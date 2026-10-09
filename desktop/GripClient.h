#pragma once
#include <QElapsedTimer>
#include <QImage>
#include <QHash>
#include <QMutex>
#include <QNetworkAccessManager>
#include <QProcess>
#include <QQuickImageProvider>
#include <QTemporaryDir>
#include <QTimer>
#include <QSoundEffect>
#include <QVariantMap>
#include <functional>

class FrameProvider final : public QQuickImageProvider {
public:
    FrameProvider() : QQuickImageProvider(QQuickImageProvider::Image) {}
    QImage requestImage(const QString &, QSize *, const QSize &) override;
    void setImage(QImage image, const QString &key = "preview");
private:
    QMutex mutex_;
    QHash<QString, QImage> images_;
};

class GripClient final : public QObject {
    Q_OBJECT
    Q_PROPERTY(bool ready READ ready NOTIFY readyChanged)
    Q_PROPERTY(QString status READ status NOTIFY statusChanged)
    Q_PROPERTY(QVariantMap schema READ schema NOTIFY schemaChanged)
    Q_PROPERTY(QVariantMap snapshot READ snapshot NOTIFY snapshotChanged)
    Q_PROPERTY(int frameRevision READ frameRevision NOTIFY frameChanged)
    Q_PROPERTY(int cropRevision READ cropRevision NOTIFY classifiedChanged)
    Q_PROPERTY(QVariantMap classifiedResult READ classifiedResult NOTIFY classifiedChanged)
    Q_PROPERTY(int artifactRevision READ artifactRevision NOTIFY artifactChanged)
    Q_PROPERTY(QString previewSession READ previewSession WRITE setPreviewSession NOTIFY previewSessionChanged)
    Q_PROPERTY(bool previewEnabled READ previewEnabled WRITE setPreviewEnabled NOTIFY previewSessionChanged)
    Q_PROPERTY(double alertVolume READ alertVolume WRITE setAlertVolume NOTIFY alertVolumeChanged)
public:
    explicit GripClient(FrameProvider *, QObject *parent = nullptr);
    void start(const QString &worker, const QStringList &workerArgs, const QString &workdir);
    Q_INVOKABLE void shutdown();
    bool ready() const { return ready_; }
    QString status() const { return status_; }
    QVariantMap schema() const { return schema_; }
    QVariantMap snapshot() const { return snapshot_; }
    int frameRevision() const { return frameRevision_; }
    int cropRevision() const { return cropRevision_; }
    QVariantMap classifiedResult() const { return classifiedResult_; }
    int artifactRevision() const { return artifactRevision_; }
    QString previewSession() const { return previewSession_; }
    bool previewEnabled() const { return previewEnabled_; }
    double alertVolume() const { return tone_.volume(); }
    void setAlertVolume(double value);
    void setPreviewSession(const QString &session);
    void setPreviewEnabled(bool enabled);
    Q_INVOKABLE void request(const QString &tag, const QString &path, const QString &method = "GET", const QVariantMap &payload = {});
    Q_INVOKABLE void run(const QString &action, const QVariantMap &payload);
    Q_INVOKABLE QString choosePath(const QString &kind, const QString &current = {});
    Q_INVOKABLE void openLocal(const QString &target);
    Q_INVOKABLE void copyText(const QString &value);
    Q_INVOKABLE void saveText(const QString &suggestedName, const QString &text);
    Q_INVOKABLE void download(const QString &path, const QString &suggestedName);
    Q_INVOKABLE void loadImage(const QString &path);
    Q_INVOKABLE void notify(bool error = false);
signals:
    void readyChanged();
    void statusChanged();
    void schemaChanged();
    void snapshotChanged();
    void frameChanged();
    void classifiedChanged();
    void artifactChanged();
    void previewSessionChanged();
    void alertVolumeChanged();
    void response(const QString &tag, const QVariantMap &payload);
    void failure(const QString &tag, const QVariantMap &payload);
private:
    using Handler = std::function<void(QByteArray, int)>;
    void send(const QString &, const QString &, const QVariantMap &, Handler);
    void poll();
    void fetchFrame();
    void fetchClassified();
    void clearClassified();
    void clearPreview();
    void setStatus(const QString &);
    FrameProvider *frames_;
    QNetworkAccessManager network_;
    QProcess worker_;
    QTemporaryDir temporary_;
    QTimer readinessTimer_, stateTimer_, frameTimer_;
    QSoundEffect tone_;
    QElapsedTimer startup_;
    QString baseUrl_, token_, status_ = "Starting the isolated processing worker…";
    QString previewSession_ = "factory", previewKey_, workdir_;
    QVariantMap schema_, snapshot_, classifiedResult_;
    bool ready_ = false, closing_ = false, stateBusy_ = false, frameBusy_ = false, previewEnabled_ = false, cropBusy_ = false;
    int frameRevision_ = 0, artifactRevision_ = 0, previewGeneration_ = 0;
    int cropRevision_ = 0, cropGeneration_ = 0;
};
