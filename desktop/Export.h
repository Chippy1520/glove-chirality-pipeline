#pragma once
#include <QIODevice>
#include <QSaveFile>

// Never replace an existing destination until the complete response is durable.
inline bool finishExport(QSaveFile &file, QIODevice &response, bool complete) {
    const auto remaining = response.readAll();
    if (!complete || file.error() != QFileDevice::NoError ||
        file.write(remaining) != remaining.size()) {
        file.cancelWriting();
        return false;
    }
    return file.commit();
}
