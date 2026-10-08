import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: inspect
    property var session: client.snapshot[client.previewSession] || ({})
    property var counters: session.counters || ({})
    property var metrics: session.metrics || ({})
    property var latest: session.latest || ({})
    property bool shadowOnly: client.previewSession === "inference"
    property string pendingMode: "shadow"
    property string checkedStartupJob: ""
    property var catalogItems: []
    property string catalogField: ""
    property bool audioEnabled: false
    property var seenEvents: ({})
    property string bootSession: ""
    spacing: 16

    function number(value) { return value === undefined || value === null ? "—" : Number(value).toFixed(1) }
    function startSession(confirmArmed) {
        const payload = Object.assign({}, setupForm.values)
        payload.confirm_armed = confirmArmed
        if (shadowOnly || payload.mode === "armed") payload.mode = "shadow"
        client.request("start-session", "/api/" + client.previewSession + "/start", "POST", payload)
    }
    onSessionChanged: {
        const job = client.previewSession + ":" + String(session.job_id || "")
        const checks = (session.preflight || {}).checks || []
        if (session.status === "fault" && job !== checkedStartupJob && checks.some(check => check.status === "failed")) {
            checkedStartupJob = job
            Qt.callLater(function() { setupForm.applyErrors(session); setupDialog.open() })
        }
    }
    onLatestChanged: {
        const id = String(session.session_id || "") + ":" + String(latest.event_id || "")
        if (bootSession !== String(session.session_id || "")) { bootSession = String(session.session_id || ""); seenEvents = {}; if (latest.event_id) seenEvents[id] = true; }
        else if (audioEnabled && latest.event_id && !seenEvents[id]) {
            seenEvents[id] = true
            if (latest.status === "accepted" && latest.prediction === (session.reject_class || "right")) client.notify(true)
        }
    }
    RowLayout {
        Label { text: "Inspection"; font.pixelSize: 28; font.bold: true; Layout.fillWidth: true; color: "#172033" }
        CheckBox { text: "Shadow-only test"; checked: inspect.shadowOnly; enabled: !inspect.session.running; onToggled: client.previewSession = checked ? "inference" : "factory" }
        Button { text: "Setup"; onClicked: setupDialog.open() }
    }
    RowLayout {
        Label { text: "Mode"; font.weight: Font.DemiBold }
        ComboBox {
            id: operatingMode
            objectName: "operatingMode"
            model: inspect.shadowOnly ? ["SHADOW"] : ["PREVIEW", "SHADOW", "ARMED"]
            currentIndex: inspect.shadowOnly ? 0 : model.indexOf(String(inspect.session.mode || setupForm.values.mode || "shadow").toUpperCase())
            enabled: client.ready && Boolean(inspect.session.running) && inspect.session.status === "running"
            onActivated: {
                inspect.pendingMode = currentText.toLowerCase()
                if (inspect.pendingMode === "armed") armDialog.open()
                else if (inspect.session.running) client.request("mode", "/api/factory/mode", "POST", {mode:inspect.pendingMode})
                else setupForm.setValue("mode", inspect.pendingMode)
            }
        }
        Label {
            text: inspect.shadowOnly ? "No physical actuation" : String(inspect.session.mode || "shadow") === "armed" ? "PHYSICAL REJECTION ENABLED" : "No physical actuation"
            color: String(inspect.session.mode || "shadow") === "armed" ? "#b42318" : "#667085"
            font.weight: Font.DemiBold
            Layout.fillWidth: true
        }
        Button { text: "Start"; highlighted: true; enabled: client.ready && !inspect.session.running; onClicked: inspect.startSession(false) }
        Button { text: "Stop"; enabled: client.ready && Boolean(inspect.session.running); onClicked: client.request("stop-session", "/api/"+client.previewSession+"/stop", "POST", {}) }
    }
    Label {
        visible: Boolean(inspect.session.fault || inspect.session.geometry_warning)
        text: inspect.session.fault || inspect.session.geometry_warning || ""
        color: "#b42318"; wrapMode: Text.Wrap; Layout.fillWidth: true
    }
    Rectangle {
        Layout.fillWidth: true
        Layout.preferredHeight: 330
        color: "#e9eef5"; radius: 10
        Image {
            anchors.fill: parent; anchors.margins: 6
            source: client.frameRevision > 0 ? "image://frames/preview?" + client.frameRevision : ""
            asynchronous: true; cache: false; fillMode: Image.PreserveAspectFit
        }
        Label { anchors.centerIn: parent; visible: client.frameRevision === 0; text: "Configure a source and model, then start in SHADOW"; color: "#667085" }
        ToolButton { anchors.right: parent.right; anchors.top: parent.top; text: "Expand"; onClicked: expandedPreview.open() }
    }
    Rectangle {
        Layout.fillWidth: true; Layout.preferredHeight: 100; radius: 10
        property bool classified: inspect.latest.status === "accepted" && Boolean(inspect.latest.prediction)
        property bool rejected: classified && inspect.latest.prediction === (inspect.session.reject_class || "right")
        color: !classified ? "#edf1f6" : rejected ? "#fdecea" : "#e8f7f0"
        Column {
            anchors.centerIn: parent; spacing: 8
            Label { anchors.horizontalCenter: parent.horizontalCenter; text: parent.parent.classified ? String(inspect.latest.prediction).toUpperCase() + (parent.parent.rejected ? " / REJECT" : " / PASS") : "WAITING"; font.pixelSize: 32; font.bold: true; color: parent.parent.rejected ? "#b42318" : "#172033" }
            Label { anchors.horizontalCenter: parent.horizontalCenter; text: parent.parent.classified ? "Confidence " + Number(inspect.latest.confidence).toFixed(3) + " · Event " + inspect.latest.event_id : String(inspect.session.status || "Idle"); color: "#667085" }
        }
    }
    RowLayout {
        Repeater {
            model: [{title:"Accepted",value:inspect.counters.total_accepted}, {title:"LEFT / PASS",value:inspect.counters.left_pass}, {title:"RIGHT / REJECT",value:inspect.counters.right_reject}, {title:"Processed FPS",value:inspect.number(inspect.metrics.processed_fps)}, {title:"Dropped frames",value:inspect.metrics.dropped_frames}]
            delegate: Rectangle {
                required property var modelData
                Layout.fillWidth: true; Layout.preferredHeight: 78; color: "white"; border.color: "#d9e1ec"; radius: 8
                Column {
                    anchors.fill: parent; anchors.margins: 12; spacing: 10
                    Label { text: modelData.title; color: "#667085"; font.pixelSize: 12 }
                    Label { text: modelData.value === undefined ? "—" : String(modelData.value); font.pixelSize: 23; font.bold: true }
                }
            }
        }
    }
    RowLayout {
        CheckBox { text: "Show preview"; checked: client.previewEnabled; onToggled: client.previewEnabled = checked }
        CheckBox { text: "Reject audio alert"; checked: inspect.audioEnabled; onToggled: inspect.audioEnabled = checked }
        Slider {from:0;to:0.5;value:client.alertVolume;Layout.preferredWidth:100;onMoved:client.alertVolume=value;Accessible.name:"Alert volume"}
        Item { Layout.fillWidth: true }
        Button { text: "Session tools"; onClicked: sessionTools.open() }
    }
    Dialog {
        id: setupDialog
        title: "Inspection setup"
        modal: true; width: Math.min(880, inspect.Window.window.width - 80); height: Math.min(720, inspect.Window.window.height - 80)
        anchors.centerIn: Overlay.overlay
        standardButtons: Dialog.Close
        ScrollView {
            anchors.fill: parent; clip: true
            ColumnLayout {
                width: parent.width - 20; spacing: 16
                Label { text: "Source/model/config/geometry changes require a stopped session. Start is non-actuating; select ARMED only after the running session passes readiness. Advanced settings retain the complete validated contract."; wrapMode: Text.Wrap; Layout.fillWidth: true; color: "#667085" }
                WorkflowForm {
                    id: setupForm
                    Layout.fillWidth: true
                    definition: ({fields:client.schema.factory || []})
                    actionsVisible: false
                    locked: Boolean(inspect.session.running)
                }
                RowLayout {
                    Button { text: "Scan cameras"; enabled: !inspect.session.running; onClicked: client.request("cameras", "/api/factory/cameras") }
                    Button { text: "Model catalog"; onClicked: client.request("checkpoints", "/api/factory/checkpoints?root="+encodeURIComponent(setupForm.values.models_root || "")) }
                    Button { text: "Config catalog"; onClicked: client.request("configs", "/api/factory/configs") }
                    Button { text: "Device readiness"; onClicked: client.request("devices", "/api/factory/devices") }
                }
                RowLayout {
                    Button { text: "Apply live policy"; enabled: Boolean(inspect.session.running); onClicked: client.request("controls", "/api/"+client.previewSession+"/controls", "POST", {decision_class:setupForm.values.decision_class,decision_threshold:Number(setupForm.values.decision_threshold),reject_class:setupForm.values.reject_class,actuator_delay_ms:Number(setupForm.values.delay_ms)}) }
                    Button { text: "Checkpoint details / SHA256"; onClicked: client.request("checkpoint", "/api/factory/checkpoint?path="+encodeURIComponent(setupForm.values.checkpoint || "")+"&sha256=1") }
                    Button { text: "Use models folder"; onClicked: client.request("models-root", "/api/factory/models-root", "POST", {path:setupForm.values.models_root || ""}) }
                }
            }
        }
    }
    Dialog {
        id: armDialog
        title: "Enable physical rejection?"
        modal: true; anchors.centerIn: Overlay.overlay
        standardButtons: Dialog.Ok | Dialog.Cancel
        Label { text: "ARMED can move machinery. Confirm shadow validation is complete, serial readiness is verified, and the actuator area is clear.\nAlready-sent firmware delays cannot be recalled."; wrapMode: Text.Wrap; width: 440; color: "#b42318" }
        onAccepted: {
            if (inspect.session.running) client.request("arm", "/api/factory/mode", "POST", {mode:"armed",confirm:true})
        }
        onClosed: operatingMode.currentIndex = Qt.binding(function() { return inspect.shadowOnly ? 0 : operatingMode.model.indexOf(String(inspect.session.mode || "shadow").toUpperCase()) })
    }
    Dialog {
        id: expandedPreview
        title: "Inspection preview"
        modal: true; anchors.centerIn: Overlay.overlay
        width: inspect.Window.window.width - 60; height: inspect.Window.window.height - 60
        standardButtons: Dialog.Close
        Image { anchors.fill: parent; source: client.frameRevision > 0 ? "image://frames/preview?"+client.frameRevision : ""; asynchronous: true; cache: false; fillMode: Image.PreserveAspectFit }
    }
    Dialog {
        id: sessionTools
        title: "Session controls and diagnostics"
        modal: true; anchors.centerIn: Overlay.overlay; width: 820; height: 650
        standardButtons: Dialog.Close
        ScrollView {
            anchors.fill: parent; clip: true
            ColumnLayout {
                width: parent.width - 20; spacing: 12
                RowLayout {
                    Button { text: "Pause"; onClicked: client.request("playback", "/api/"+client.previewSession+"/playback", "POST", {pause:true}) }
                    Button { text: "Resume"; onClicked: client.request("playback", "/api/"+client.previewSession+"/playback", "POST", {pause:false}) }
                    ComboBox { model: ["0.25", "0.5", "1", "2"]; currentIndex: 2; onActivated: client.request("playback", "/api/"+client.previewSession+"/playback", "POST", {speed:Number(currentText)}) }
                    Button { text: "Reset counters"; onClicked: resetCounters.open() }
                }
                RowLayout {
                    Button { text: "Serial ports"; enabled: !inspect.shadowOnly; onClicked: client.request("ports", "/api/factory/ports") }
                    Button { text: "Connect serial"; enabled: !inspect.shadowOnly; onClicked: client.request("serial", "/api/factory/serial/connect", "POST", {port:setupForm.values.port,baud:Number(setupForm.values.baud)}) }
                    Button { text: "Disconnect"; enabled: !inspect.shadowOnly; onClicked: client.request("serial", "/api/factory/serial/disconnect", "POST", {}) }
                    Button { text: "Physical test…"; enabled: !inspect.shadowOnly && Boolean(inspect.session.running) && inspect.session.mode === "armed" && Boolean((inspect.session.serial || {}).connected); onClicked: physicalTest.open() }
                }
                RowLayout {
                    Button { text: "Show size-rejected boxes"; onClicked: client.request("display", "/api/"+client.previewSession+"/display", "POST", {show_size_rejected:true}) }
                    Button { text: "Hide rejected boxes"; onClicked: client.request("display", "/api/"+client.previewSession+"/display", "POST", {show_size_rejected:false}) }
                    Button { text: "Simulate alarm (no actuator)"; enabled: inspect.shadowOnly; onClicked: client.request("simulate", "/api/inference/simulate-trigger", "POST", {}) }
                }
                RowLayout {
                    Button { text:"Export CSV";onClicked:client.download("/api/"+client.previewSession+"/export?format=csv","passages.csv") }
                    Button { text:"Export JSONL";onClicked:client.download("/api/"+client.previewSession+"/export?format=jsonl","passages.jsonl") }
                    Button { text:"Latest canonical crop";onClicked:client.loadImage("/api/"+client.previewSession+"/crop.jpg") }
                }
                TextArea { Layout.fillWidth: true; text: JSON.stringify(inspect.session,null,2); readOnly: true; wrapMode: Text.WrapAnywhere; font.family: "Consolas"; font.pixelSize: 12 }
            }
        }
    }
    Dialog {
        id: resetCounters
        title: "Reset session counters?"
        modal: true; anchors.centerIn: Overlay.overlay
        standardButtons: Dialog.Ok | Dialog.Cancel
        Label { width: 440; wrapMode: Text.Wrap; text: "Reset displayed counters for this session? Existing event logs remain unchanged." }
        onAccepted: client.request("reset", "/api/"+client.previewSession+"/counters/reset", "POST", {confirm:true})
    }
    Dialog {
        id: physicalTest
        title: "Physical actuator test"
        modal: true; anchors.centerIn: Overlay.overlay
        standardButtons: Dialog.Ok | Dialog.Cancel
        Label { width: 440; wrapMode: Text.Wrap; text: "Confirm attended ARMED testing and that the actuator area is clear. This sends a physical trigger."; color: "#b42318" }
        onAccepted: client.request("physical-test", "/api/factory/manual-trigger", "POST", {confirm:true})
    }
    Connections {
        target: client
        function onFailure(tag, report) { if (tag === "start-session") { setupForm.applyErrors(report); setupDialog.open() } }
        function onResponse(tag,payload) {
            if (tag === "cameras") { inspect.catalogField="source";inspect.catalogItems=(payload.cameras || []).map(item=>({label:item.label,value:String(item.index)}));catalog.open() }
            if (tag === "checkpoints" || tag === "models-root") { inspect.catalogField="checkpoint";inspect.catalogItems=(payload.checkpoints || []).map(item=>({label:item.label,value:item.path}));catalog.open() }
            if (tag === "configs") { inspect.catalogField="config";inspect.catalogItems=(payload.configs || []).map(item=>({label:item.name,value:item.path}));catalog.open() }
        }
    }
    Dialog {
        id:catalog
        title:"Select " + inspect.catalogField
        modal:true;anchors.centerIn:Overlay.overlay;width:720;height:440
        standardButtons:Dialog.Close
        ListView {
            anchors.fill:parent;clip:true;spacing:6
            model:inspect.catalogItems
            delegate:Button {
                required property var modelData
                width:ListView.view.width;text:modelData.label
                onClicked:{setupForm.setValue(inspect.catalogField,modelData.value);catalog.close()}
            }
        }
    }
}
