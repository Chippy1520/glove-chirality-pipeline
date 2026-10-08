#include "GripClient.h"
#include <QApplication>
#include <QClipboard>
#include <QDesktopServices>
#include <QDataStream>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QFutureWatcher>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMutexLocker>
#include <QNetworkReply>
#include <QUrl>
#include <QtConcurrent>
#include <cmath>

QImage FrameProvider::requestImage(const QString &id, QSize *size, const QSize &requested) {
    QMutexLocker lock(&mutex_);
    const auto image = images_.value(id.section('?', 0, 0));
    if (size) *size = image.size();
    return requested.isValid() ? image.scaled(requested, Qt::KeepAspectRatio, Qt::SmoothTransformation) : image;
}
void FrameProvider::setImage(QImage image, const QString &key) {
    QMutexLocker lock(&mutex_);
    images_.insert(key, std::move(image));
}

GripClient::GripClient(FrameProvider *frames, QObject *parent) : QObject(parent), frames_(frames) {
    QFile audio(temporary_.filePath("alert.wav"));
    if (audio.open(QIODevice::WriteOnly)) {
        QDataStream data(&audio); data.setByteOrder(QDataStream::LittleEndian);
        constexpr int rate = 16000, samples = 2400, bytes = samples * 2;
        data.writeRawData("RIFF",4); data << quint32(36+bytes); data.writeRawData("WAVEfmt ",8);
        data << quint32(16) << quint16(1) << quint16(1) << quint32(rate) << quint32(rate*2) << quint16(2) << quint16(16);
        data.writeRawData("data",4); data << quint32(bytes);
        for (int i=0;i<samples;++i) data << qint16(12000 * std::sin(6.283185307179586 * 880 * i / rate) * std::sin(3.141592653589793 * i / samples));
        audio.close(); tone_.setSource(QUrl::fromLocalFile(audio.fileName())); tone_.setVolume(0.08);
    }
    network_.setTransferTimeout(8000);
    readinessTimer_.setInterval(100);
    stateTimer_.setInterval(1000);
    frameTimer_.setInterval(100);
    connect(&stateTimer_, &QTimer::timeout, this, &GripClient::poll);
    connect(&frameTimer_, &QTimer::timeout, this, &GripClient::fetchFrame);
    connect(&worker_, &QProcess::errorOccurred, this, [this](QProcess::ProcessError) {
        setStatus("Worker could not start: " + worker_.errorString());
        ready_ = false; emit readyChanged();
    });
    connect(&worker_, &QProcess::finished, this, [this](int code, QProcess::ExitStatus) {
        readinessTimer_.stop(); stateTimer_.stop(); frameTimer_.stop();
        ready_ = false; emit readyChanged();
        setStatus(closing_ ? "Stopped safely" : QString("Processing worker stopped (%1). Inspection is unavailable.").arg(code));
        if (closing_) QCoreApplication::quit();
    });
}
void GripClient::setStatus(const QString &value) {
    if (status_ == value) return;
    status_ = value; emit statusChanged();
}
void GripClient::start(const QString &worker, const QStringList &workerArgs, const QString &workdir) {
    const QString readyFile = temporary_.filePath("ready.json");
    QStringList args = workerArgs;
    args << "serve" << "--workdir" << workdir << "--ready-file" << readyFile;
    worker_.setWorkingDirectory(workdir);
    worker_.setProgram(worker);
    worker_.setArguments(args);
    worker_.setProcessChannelMode(QProcess::SeparateChannels);
    connect(&worker_, &QProcess::readyReadStandardOutput, this, [this] { worker_.readAllStandardOutput(); });
    connect(&worker_, &QProcess::readyReadStandardError, this, [this] {
        const auto bytes = worker_.readAllStandardError();
        if (!ready_ && !bytes.isEmpty()) setStatus(QString::fromUtf8(bytes).right(500));
    });
    connect(&readinessTimer_, &QTimer::timeout, this, [this, readyFile] {
        QFile file(readyFile);
        if (!file.open(QIODevice::ReadOnly)) {
            if (startup_.elapsed() > 60000) { readinessTimer_.stop(); setStatus("Worker startup timed out; inspect the deployment diagnostics."); worker_.kill(); }
            return;
        }
        const auto data = QJsonDocument::fromJson(file.readAll()).object();
        file.close();
        const QUrl endpoint(data.value("url").toString());
        if (endpoint.scheme() != "http" || endpoint.host() != "127.0.0.1" || data.value("token").toString().isEmpty()) return;
        baseUrl_ = endpoint.toString(); token_ = data.value("token").toString();
        QFile::remove(readyFile); readinessTimer_.stop();
        ready_ = true; emit readyChanged(); setStatus("Ready · SHADOW is the default");
        request("schema", "/api/desktop/schema");
        poll(); stateTimer_.start(); frameTimer_.start();
    });
    startup_.start(); worker_.start(); readinessTimer_.start();
}
void GripClient::send(const QString &path, const QString &method, const QVariantMap &payload, Handler handler) {
    if (!ready_ || !path.startsWith("/api/") || path.startsWith("//") || path.contains("://")) return;
    QNetworkRequest request(QUrl(baseUrl_ + path));
    request.setRawHeader("X-GRIP-Desktop", token_.toUtf8());
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::ManualRedirectPolicy);
    QNetworkReply *reply = nullptr;
    if (method == "GET") reply = network_.get(request);
    else reply = network_.sendCustomRequest(request, method.toUtf8(), QJsonDocument::fromVariant(payload).toJson(QJsonDocument::Compact));
    connect(reply, &QNetworkReply::finished, this, [reply, handler] {
        const auto status = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
        const auto bytes = reply->readAll();
        handler(bytes, status); reply->deleteLater();
    });
}
void GripClient::request(const QString &tag, const QString &path, const QString &method, const QVariantMap &payload) {
    if (!ready_ || closing_) return;
    send(path, method, payload, [this, tag](QByteArray bytes, int code) {
        auto data = QJsonDocument::fromJson(bytes).toVariant().toMap();
        if (code < 200 || code >= 300) {
            if (data.isEmpty()) data.insert("error", code ? QString("Request failed (%1)").arg(code) : "Worker connection lost. Displayed state may be stale.");
            setStatus(data.value("error").toString()); emit failure(tag, data); return;
        }
        if (tag == "schema") { schema_ = data; emit schemaChanged(); }
        emit response(tag, data);
    });
}
void GripClient::run(const QString &action, const QVariantMap &payload) {
    auto data = payload; data.insert("action", action);
    request(action, "/api/run", "POST", data);
}
void GripClient::poll() {
    if (!ready_ || stateBusy_ || closing_) return;
    stateBusy_ = true;
    send("/api/desktop/state", "GET", {}, [this](QByteArray bytes, int code) {
        stateBusy_ = false;
        if (code != 200) { setStatus("Backend connection lost · displayed state may be stale"); return; }
        snapshot_ = QJsonDocument::fromJson(bytes).toVariant().toMap(); emit snapshotChanged();
    });
}
void GripClient::setPreviewSession(const QString &session) {
    if (session != "factory" && session != "inference") return;
    if (previewSession_ == session) return;
    previewSession_ = session; emit previewSessionChanged();
}
void GripClient::setPreviewEnabled(bool enabled) {
    if (previewEnabled_ == enabled) return;
    previewEnabled_ = enabled; emit previewSessionChanged();
}
void GripClient::fetchFrame() {
    if (!ready_ || frameBusy_ || !previewEnabled_ || closing_) return;
    if (!snapshot_.value(previewSession_).toMap().value("running").toBool()) return;
    frameBusy_ = true;
    send("/api/" + previewSession_ + "/frame.jpg", "GET", {}, [this](QByteArray bytes, int code) {
        if (code != 200 || bytes.isEmpty()) { frameBusy_ = false; return; }
        auto *watcher = new QFutureWatcher<QImage>(this);
        connect(watcher, &QFutureWatcher<QImage>::finished, this, [this, watcher] {
            const auto image = watcher->result(); watcher->deleteLater(); frameBusy_ = false;
            if (image.isNull()) return;
            frames_->setImage(image); ++frameRevision_; emit frameChanged();
        });
        watcher->setFuture(QtConcurrent::run([bytes] { return QImage::fromData(bytes); }));
    });
}
QString GripClient::choosePath(const QString &kind, const QString &current) {
    // Qt's dialog keeps timers/heartbeats running; Windows' static native dialog
    // can suspend QTimer delivery while an operator chooses a path.
    const auto options = QFileDialog::DontUseNativeDialog;
    if (kind == "directory") return QFileDialog::getExistingDirectory(nullptr, "Choose folder", current, options);
    if (kind == "save") return QFileDialog::getSaveFileName(nullptr, "Save output", current, {}, nullptr, options);
    return QFileDialog::getOpenFileName(nullptr, "Choose file", current, {}, nullptr, options);
}
void GripClient::openLocal(const QString &target) {
    const QUrl url(target);
    if ((url.scheme() == "http" || url.scheme() == "https") && (url.host() == "127.0.0.1" || url.host() == "localhost")) QDesktopServices::openUrl(url);
    else if (url.scheme().isEmpty() && QFileInfo::exists(target)) QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo(target).absoluteFilePath()));
}
void GripClient::copyText(const QString &value) { QApplication::clipboard()->setText(value); }
void GripClient::saveText(const QString &suggestedName, const QString &text) {
    const auto path = QFileDialog::getSaveFileName(nullptr, "Export", suggestedName, {}, nullptr, QFileDialog::DontUseNativeDialog);
    if (path.isEmpty()) return;
    QFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(text.toUtf8()) < 0)
        emit failure("export", {{"error", "Could not write the selected output file"}});
}
void GripClient::setAlertVolume(double value) { tone_.setVolume(qBound(0.0, value, 1.0)); emit alertVolumeChanged(); }
void GripClient::notify(bool) { if (!tone_.isPlaying()) tone_.play(); }
void GripClient::download(const QString &path, const QString &suggestedName) {
    const auto target = QFileDialog::getSaveFileName(nullptr, "Export", suggestedName, {}, nullptr, QFileDialog::DontUseNativeDialog);
    if (target.isEmpty()) return;
    if (!ready_ || !path.startsWith("/api/") || path.contains("://")) return;
    QNetworkRequest request(QUrl(baseUrl_ + path));
    request.setRawHeader("X-GRIP-Desktop", token_.toUtf8());
    request.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::ManualRedirectPolicy);
    auto *reply = network_.get(request);
    auto *file = new QFile(target, reply);
    if (!file->open(QIODevice::WriteOnly)) { reply->abort(); reply->deleteLater(); emit failure("download", {{"error", "Could not open the selected output file"}}); return; }
    connect(reply, &QNetworkReply::readyRead, this, [reply, file] {
        if (reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200) file->write(reply->readAll());
    });
    connect(reply, &QNetworkReply::finished, this, [this, reply, file] {
        const bool ok = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt() == 200 && file->error() == QFileDevice::NoError;
        file->close();
        if (!ok) { file->remove(); emit failure("download", {{"error", "Export failed; partial output removed"}}); }
        reply->deleteLater();
    });
}
void GripClient::loadImage(const QString &path) {
    send(path, "GET", {}, [this](QByteArray bytes, int code) {
        if (code != 200) { emit failure("image", {{"error", "Image unavailable; start a session or select an existing artifact"}}); return; }
        auto *watcher = new QFutureWatcher<QImage>(this);
        connect(watcher, &QFutureWatcher<QImage>::finished, this, [this, watcher] {
            const auto image = watcher->result(); watcher->deleteLater();
            if (image.isNull()) { emit failure("image", {{"error", "Could not decode the returned image"}}); return; }
            frames_->setImage(image, "artifact"); ++artifactRevision_; emit artifactChanged();
        });
        watcher->setFuture(QtConcurrent::run([bytes] { return QImage::fromData(bytes); }));
    });
}
void GripClient::shutdown() {
    if (closing_) return;
    closing_ = true; stateTimer_.stop(); frameTimer_.stop(); readinessTimer_.stop();
    setStatus("Stopping inspection and owned jobs…");
    if (ready_) send("/api/desktop/shutdown", "POST", {}, [](QByteArray, int) {});
    else if (worker_.state() == QProcess::NotRunning) { QCoreApplication::quit(); return; }
    QTimer::singleShot(7000, this, [this] { worker_.kill(); QTimer::singleShot(500, qApp, &QCoreApplication::quit); });
}
