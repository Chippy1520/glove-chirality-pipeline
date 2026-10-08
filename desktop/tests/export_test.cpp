#include "Export.h"
#include <QBuffer>
#include <QCoreApplication>
#include <QFile>
#include <QTemporaryDir>

static void require(bool condition, const char *message) {
    if (!condition) qFatal("%s", message);
}
static QByteArray read(const QString &path) {
    QFile file(path);
    require(file.open(QIODevice::ReadOnly), "read destination");
    return file.readAll();
}
int main(int argc, char **argv) {
    QCoreApplication app(argc, argv);
    QTemporaryDir directory;
    require(directory.isValid(), "temporary directory");
    const auto path = directory.filePath("existing.csv");
    QFile original(path);
    require(original.open(QIODevice::WriteOnly), "create destination");
    require(original.write("original") == 8, "write original");
    original.close();
    {
        QSaveFile output(path);
        require(output.open(QIODevice::WriteOnly), "open failed download");
        require(output.write("partial") == 7, "stream partial download");
        QBuffer response; response.setData("truncated"); response.open(QIODevice::ReadOnly);
        require(!finishExport(output, response, false), "interrupted network must fail");
        require(read(path) == "original", "interrupted export must preserve existing data");
    }
    {
        QSaveFile output(path);
        require(output.open(QIODevice::WriteOnly), "open successful download");
        require(output.write("first,") == 6, "stream successful download");
        QBuffer response; response.setData("last\n"); response.open(QIODevice::ReadOnly);
        require(finishExport(output, response, true), "complete export must commit");
        require(read(path) == "first,last\n", "final unread bytes must be exported");
    }
    qInfo("PASS: interrupted export preserves destination; complete export drains and commits");
}
