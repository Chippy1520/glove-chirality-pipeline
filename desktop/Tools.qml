import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: tools
    property string resultText: ""
    property var comparison: []
    property var definitions: client.schema.forms || []
    function form(action) { return definitions.find(item => item.action === action) || ({fields:[]}) }
    spacing: 16
    Label { text: "Advanced workspace"; font.pixelSize: 28; font.bold: true; color: "#172033" }
    Label { text: "Calibration, configuration and research tools remain available without cluttering inspection."; color: "#667085"; wrapMode: Text.Wrap; Layout.fillWidth: true }
    ComboBox {
        id: choice
        objectName: "advancedTool"
        Layout.fillWidth: true
        model: ["Layer 1 configuration", "Calibration preview", "Model comparison", "Explain a prediction", "TensorBoard", "Tasks, logs and artifacts", "Read-only LAN sharing"]
    }
    ColumnLayout {
        visible: choice.currentIndex === 0
        Layout.fillWidth: true
        RowLayout {
            TextField { id: configPath; Layout.fillWidth: true; text: "configs/factory.yaml"; placeholderText: "Configuration YAML" }
            Button { text: "Browse"; onClicked: {const value=client.choosePath("file",configPath.text);if(value)configPath.text=value} }
            Button { text: "Load"; onClicked: client.request("load-config", "/api/config?path="+encodeURIComponent(configPath.text)) }
            Button { text: "Validate & save"; highlighted: true; onClicked: client.request("save-config", "/api/config", "PUT", {path:configPath.text,text:yamlEditor.text}) }
        }
        TextArea {
            id: yamlEditor
            objectName: "yamlEditor"
            Layout.fillWidth: true
            Layout.preferredHeight: 330
            selectByMouse: true; wrapMode: Text.NoWrap
            font.family: "Consolas"; font.pixelSize: 13
            placeholderText: "Load a configuration. Existing validation, unknown-key checks and atomic replacement are retained."
        }
        RowLayout {
            TextField { id: yoloPath; Layout.fillWidth: true; placeholderText: "YOLO segmentation checkpoint for preset" }
            Button { text: "Browse"; onClicked: {const value=client.choosePath("file",yoloPath.text);if(value)yoloPath.text=value} }
            Button { text: "YOLO preset"; onClicked: client.request("preset", "/api/config/preset", "POST", {path:configPath.text,preset:"custom_yolo",model:yoloPath.text}) }
            Button { text: "Tight crop preset"; onClicked: client.request("preset", "/api/config/preset", "POST", {path:configPath.text,preset:"tight_crop"}) }
        }
        Button { text: "Record geometry review (not certification)"; onClicked: tools.resultText = "Geometry reviewed locally. This is not production certification; validate a generated preview and real-domain extraction." }
    }
    WorkflowForm { Layout.fillWidth: true; visible: choice.currentIndex === 1; definition: tools.form("preview"); buttonLabel:"Render calibration preview"; onSubmitted:payload=>client.run("preview",payload) }
    ColumnLayout {
        visible: choice.currentIndex === 2
        Layout.fillWidth: true
        RowLayout {
            TextField { id: comparisonRoot; Layout.fillWidth: true; placeholderText: "Metrics root folder"; text: client.snapshot.comparison_root || "" }
            Button { text: "Browse"; onClicked: {const value=client.choosePath("directory",comparisonRoot.text);if(value)comparisonRoot.text=value} }
            Button { text: "Use folder"; onClicked: client.request("comparison-root", "/api/comparison/root", "PUT", {path:comparisonRoot.text}) }
        }
        RowLayout {
            ComboBox { id: metric; model: client.schema.comparison_metrics || []; Layout.fillWidth: true }
            Button { text: "Refresh comparison"; onClicked: client.request("comparison", "/api/comparison?metric="+encodeURIComponent(metric.currentText || "recall_right")) }
            Button { text: "Export comparison CSV"; onClicked: client.saveText("model_comparison.csv",tools.comparisonCsv()) }
        }
        Repeater {
            model: tools.comparison
            delegate: Label {
                required property var modelData
                Layout.fillWidth: true; wrapMode: Text.Wrap
                text: (modelData.model_name || modelData.model || "Model") + " · " + JSON.stringify(modelData.metrics || modelData.best_validation || modelData)
            }
        }
        Label { text:"Rank only comparable splits and protocols; historical validation is not factory test accuracy."; color:"#667085"; wrapMode:Text.Wrap;Layout.fillWidth:true }
    }
    WorkflowForm { Layout.fillWidth:true;visible:choice.currentIndex===3;definition:tools.form("explain");buttonLabel:"Generate explanation";onSubmitted:payload=>client.run("explain",payload) }
    ColumnLayout {
        visible:choice.currentIndex===4;Layout.fillWidth:true
        WorkflowForm { Layout.fillWidth:true;definition:tools.form("tensorboard");buttonLabel:"Start TensorBoard";onSubmitted:payload=>client.run("tensorboard",payload) }
        RowLayout {
            Button { text:"Open dashboard";onClicked:client.openLocal("http://127.0.0.1:6006") }
            Button { text:"Stop TensorBoard";onClicked:client.request("stop-tensorboard","/api/stop/tensorboard","POST",{}) }
        }
    }
    ColumnLayout {
        visible:choice.currentIndex===5;Layout.fillWidth:true
        Button { text:"Refresh complete diagnostics";onClicked:client.request("diagnostics","/api/state") }
        Repeater {
            model:Object.values(client.snapshot.jobs || {}).filter(item=>item)
            delegate:ColumnLayout {
                required property var modelData
                Layout.fillWidth:true
                Label { text:(modelData.action || "Workflow")+" · "+(modelData.status || "")+" · "+(modelData.stage || "");font.bold:true }
                RowLayout {
                    Button { text:"Details";onClicked:client.request("job","/api/jobs/"+encodeURIComponent(modelData.job_id)) }
                    Button { text:"Log";onClicked:client.request("job-log","/api/jobs/"+encodeURIComponent(modelData.job_id)+"/logs") }
                    Button { text:"Artifacts";onClicked:client.request("artifacts","/api/jobs/"+encodeURIComponent(modelData.job_id)+"/artifacts") }
                    Button { text:"Cancel";onClicked:client.request("cancel","/api/jobs/"+encodeURIComponent(modelData.job_id)+"/stop","POST",{}) }
                    Button { text:"Open output";enabled:Boolean(modelData.output);onClicked:client.openLocal(modelData.output) }
                }
                TextArea { Layout.fillWidth:true;text:JSON.stringify(modelData.result || {},null,2);readOnly:true;wrapMode:Text.WrapAnywhere;font.family:"Consolas";font.pixelSize:12 }
            }
        }
    }
    ColumnLayout {
        visible:choice.currentIndex===6;Layout.fillWidth:true
        Label { text:"Opt-in sharing on a trusted private network only. Viewer has no mutation, file-browser, raw-log, model-loading or actuator routes.";wrapMode:Text.Wrap;Layout.fillWidth:true;color:"#667085" }
        CheckBox { id:trusted; text:"I confirm this is a trusted private network" }
        Button { text:"Enable read-only viewer";enabled:trusted.checked;onClicked:client.request("sharing","/api/desktop/sharing","POST",{confirm_private_network:true}) }
        Button { text:"Copy displayed viewer link";onClicked:client.copyText(tools.resultText) }
        Button { text:"Show viewer QR";onClicked:client.loadImage("/api/lan/qr.png") }
    }
    TextArea {
        Layout.fillWidth:true
        Layout.preferredHeight:180
        text:tools.resultText
        readOnly:true;selectByMouse:true;wrapMode:Text.WrapAnywhere
        font.family:"Consolas";font.pixelSize:12
        placeholderText:"Results, startup checks and diagnostic responses appear here."
    }
    function comparisonCsv() {
        const escaped = value=>'"'+String(value===undefined?"":value).replace(/"/g,'""')+'"'
        return "model,result\n" + tools.comparison.map(item=>escaped(item.model_name || item.model || "")+","+escaped(JSON.stringify(item))).join("\n")
    }
    Connections {
        target:client
        function onResponse(tag,payload) {
            if(tag==="load-config" || tag==="preset")yamlEditor.text=payload.text || ""
            if(tag==="comparison")tools.comparison=payload.runs || []
            if(tag!=="schema")tools.resultText=tag==="sharing" ? payload.url || "" : JSON.stringify(payload,null,2)
        }
        function onFailure(tag,payload) {tools.resultText=JSON.stringify(payload,null,2)}
    }
}
