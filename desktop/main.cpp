#include "GripClient.h"
#include <QApplication>
#include <QCommandLineParser>
#include <QDir>
#include <QFile>
#include <QQuickWindow>
#include <QQuickStyle>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include <QStandardPaths>
#include <QTimer>

int main(int argc, char **argv) {
    QApplication app(argc, argv);
    QQuickStyle::setStyle("Fusion");
    app.setApplicationName("GRIP"); app.setOrganizationName("GRIP Research");
    app.setApplicationVersion("0.1.0");
    const auto logFolder = QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation);
    QDir().mkpath(logFolder);
    qInstallMessageHandler([](QtMsgType, const QMessageLogContext &, const QString &message) {
        QFile log(QStandardPaths::writableLocation(QStandardPaths::AppLocalDataLocation) + "/native.log");
        if (log.open(QIODevice::WriteOnly | QIODevice::Append)) log.write(message.toUtf8() + '\n');
    });
    QCommandLineParser parser; parser.addHelpOption(); parser.addVersionOption();
    parser.addOption({"workspace", "Folder containing configs, data and model paths.", "path"});
    parser.addOption({"backend-python", "Development-only Python interpreter (deployed build uses its worker executable).", "path"});
    parser.addOption({"capture-directory", "Capture native acceptance screenshots and exit without starting inspection.", "path"});
    parser.process(app);
    QString workdir = parser.value("workspace");
    if (workdir.isEmpty()) workdir = QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/GRIP";
    workdir = QDir(workdir).absolutePath(); QDir().mkpath(workdir);
    QDir sourceConfigs(QCoreApplication::applicationDirPath() + "/configs");
    if (sourceConfigs.exists()) {
        QDir().mkpath(workdir + "/configs");
        for (const auto &name : sourceConfigs.entryList({"*.yaml"}, QDir::Files))
            if (!QFile::exists(workdir + "/configs/" + name)) QFile::copy(sourceConfigs.filePath(name), workdir + "/configs/" + name);
    }
    auto *provider = new FrameProvider;
    GripClient client(provider);
    QQmlApplicationEngine engine;
    engine.addImageProvider("frames", provider);
    engine.rootContext()->setContextProperty("client", &client);
    QObject::connect(&engine, &QQmlApplicationEngine::objectCreationFailed, &app, [] { QCoreApplication::exit(1); }, Qt::QueuedConnection);
    engine.loadFromModule("GRIP", "Main");
    if (engine.rootObjects().isEmpty()) return 1;
    const auto python = parser.value("backend-python");
    const auto worker = python.isEmpty() ? QCoreApplication::applicationDirPath() + "/worker/grip-worker.exe" : python;
    const QStringList workerArgs = python.isEmpty() ? QStringList{} : QStringList{"-m", "glove_chirality.desktop_worker"};
    client.start(worker, workerArgs, workdir);
    const auto captures = parser.value("capture-directory");
    if (!captures.isEmpty()) {
        QDir().mkpath(captures);
        QObject::connect(&client, &GripClient::schemaChanged, &app, [&engine, &client, captures] {
            auto *window = qobject_cast<QQuickWindow *>(engine.rootObjects().first());
            if (!window) return;
            for (int index = 0; index < 4; ++index) {
                QTimer::singleShot(600 + index * 900, window, [window, captures, index] {
                    window->setProperty("currentArea", index);
                    QTimer::singleShot(400, window, [window, captures, index] {
                        window->grabWindow().save(captures + QString("/native_%1.png").arg(index));
                    });
                });
            }
            QTimer::singleShot(5000, &client, &GripClient::shutdown);
        });
        QTimer::singleShot(65000, &client, &GripClient::shutdown);
    }
    return app.exec();
}
