import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: window
    objectName: "GRIPWindow"
    width: 1280; height: 900
    minimumWidth: 1024; minimumHeight: 720
    visible: true
    title: "GRIP · Native inspection workstation"
    color: "#f3f6fa"
    palette.window: "#f3f6fa"
    palette.windowText: "#172033"
    palette.button: "#ffffff"
    palette.buttonText: "#172033"
    palette.base: "#ffffff"
    palette.text: "#172033"
    palette.highlight: "#2457d6"
    palette.highlightedText: "#ffffff"
    palette.placeholderText: "#667085"
    font.family: "Segoe UI"
    font.pixelSize: 14
    property int currentArea: 0
    property var dataActions: ["extract_dataset","extract_single","audit_dataset"]
    property var modelActions: ["train","infer_video","infer_images","infer_live"]
    property var forms: client.schema.forms || []
    function definition(action) {return forms.find(item=>item.action===action) || ({fields:[]})}
    onClosing:close=>{close.accepted=false;client.shutdown()}
    onCurrentAreaChanged:client.previewEnabled=currentArea===0
    header: Rectangle {
        height: 78; color: "#101828"
        RowLayout {
            anchors.fill:parent;anchors.margins:20;spacing:20
            Label {text:"GRIP";font.pixelSize:24;font.bold:true;color:"white"}
            Label {text:"Native vision workstation";color:"#b7c6dc";Layout.fillWidth:true}
            Label {text:client.ready?"HOST CONTROLS":"CONNECTING";color:"#89b4ff";font.weight:Font.DemiBold}
        }
    }
    RowLayout {
        anchors.fill:parent;spacing:0
        Rectangle {
            Layout.fillHeight:true;Layout.preferredWidth:185;color:"white";border.color:"#d9e1ec"
            ColumnLayout {
                anchors.fill:parent;anchors.margins:16;spacing:8
                Repeater {
                    model:["Inspect","Data","Models","Advanced"]
                    delegate:Button {
                        required property int index
                        required property string modelData
                        Layout.fillWidth:true;text:modelData
                        highlighted:window.currentArea===index
                        onClicked:window.currentArea=index
                    }
                }
                Item {Layout.fillHeight:true}
                Label {text:"FP32 baseline\nShared extractor\nSource-isolated learning";color:"#667085";font.pixelSize:12;lineHeight:1.4;Layout.fillWidth:true;wrapMode:Text.Wrap}
            }
        }
        ColumnLayout {
            Layout.fillWidth:true;Layout.fillHeight:true;spacing:0
            Rectangle {
                Layout.fillWidth:true;Layout.preferredHeight:48;color:client.ready?"#eaf0fc":"#fff3e5"
                Label {anchors.fill:parent;anchors.margins:12;text:client.status;wrapMode:Text.Wrap;elide:Text.ElideRight;color:"#344054";font.pixelSize:12}
            }
            ScrollView {
                id:workspace
                Layout.fillWidth:true;Layout.fillHeight:true;clip:true
                ScrollBar.horizontal.policy:ScrollBar.AlwaysOff
                ColumnLayout {
                    width:workspace.availableWidth-48
                    x:24;y:24;spacing:20
                    Inspect {Layout.fillWidth:true;visible:window.currentArea===0}
                    ColumnLayout {
                        Layout.fillWidth:true;visible:window.currentArea===1;spacing:16
                        Label {text:"Prepare and audit data";font.pixelSize:28;font.bold:true;color:"#172033"}
                        Label {text:"One shared passage extractor. Source labels attach after crop selection.";color:"#667085";Layout.fillWidth:true;wrapMode:Text.Wrap}
                        ComboBox {id:dataAction;Layout.fillWidth:true;model:window.dataActions.map(action=>window.definition(action).title || action)}
                        WorkflowForm {
                            id:dataForm;Layout.fillWidth:true
                            definition:window.definition(window.dataActions[dataAction.currentIndex])
                            buttonLabel:"Run data workflow"
                            onSubmitted:payload=>client.run(window.dataActions[dataAction.currentIndex],payload)
                        }
                    }
                    ColumnLayout {
                        Layout.fillWidth:true;visible:window.currentArea===2;spacing:16
                        Label {text:"Train and evaluate models";font.pixelSize:28;font.bold:true;color:"#172033"}
                        Label {text:"Keep source groups separate. RIGHT→LEFT errors remain safety-critical.";color:"#667085";Layout.fillWidth:true;wrapMode:Text.Wrap}
                        ComboBox {id:modelAction;Layout.fillWidth:true;model:window.modelActions.map(action=>window.definition(action).title || action)}
                        WorkflowForm {
                            id:modelForm;Layout.fillWidth:true
                            definition:window.definition(window.modelActions[modelAction.currentIndex])
                            buttonLabel:"Run model workflow"
                            onSubmitted:payload=>client.run(window.modelActions[modelAction.currentIndex],payload)
                        }
                        Button {
                            visible:window.modelActions[modelAction.currentIndex]==="train"
                            text:"Validate dataset and settings"
                            onClicked:{const payload=Object.assign({action:"train"},modelForm.values);client.request("preflight","/api/jobs/preflight","POST",payload)}
                        }
                        TextArea {
                            visible:Object.values(client.snapshot.jobs || {}).some(item=>item)
                            Layout.fillWidth:true;text:JSON.stringify(client.snapshot.jobs || {},null,2)
                            readOnly:true;wrapMode:Text.WrapAnywhere;font.family:"Consolas";font.pixelSize:12
                        }
                    }
                    Tools {Layout.fillWidth:true;visible:window.currentArea===3}
                    Item {Layout.preferredHeight:24}
                }
            }
        }
    }
    Dialog {
        id:imageDialog
        title:"Canonical crop / calibration / explanation / sharing"
        modal:true;anchors.centerIn:Overlay.overlay;width:900;height:660
        standardButtons:Dialog.Close
        Image {anchors.fill:parent;source:client.artifactRevision>0?"image://frames/artifact?"+client.artifactRevision:"";asynchronous:true;cache:false;fillMode:Image.PreserveAspectFit}
    }
    Dialog {
        id:resultDialog
        title:"Resource result"
        modal:true;anchors.centerIn:Overlay.overlay;width:760;height:560
        standardButtons:Dialog.Close
        property var payload: ({})
        ScrollView {anchors.fill:parent;TextArea {text:JSON.stringify(resultDialog.payload,null,2);readOnly:true;selectByMouse:true;wrapMode:Text.WrapAnywhere;font.family:"Consolas";font.pixelSize:13}}
    }
    Dialog {
        id:failureDialog
        title:"Workflow needs attention"
        modal:true;anchors.centerIn:Overlay.overlay;width:760;height:500
        standardButtons:Dialog.Close
        property var report: ({})
        ColumnLayout {
            anchors.fill:parent
            Label {text:typeof failureDialog.report.error==="object"?JSON.stringify(failureDialog.report.error):String(failureDialog.report.error || "Request failed");color:"#b42318";wrapMode:Text.Wrap;Layout.fillWidth:true;font.bold:true}
            ScrollView {Layout.fillWidth:true;Layout.fillHeight:true;TextArea {text:JSON.stringify(failureDialog.report,null,2);readOnly:true;wrapMode:Text.WrapAnywhere;font.family:"Consolas";font.pixelSize:12}}
            Button {text:"Open advanced diagnostics";onClicked:{window.currentArea=3;failureDialog.close()}}
        }
    }
    Connections {
        target:client
        function onArtifactChanged() {imageDialog.open()}
        function onFailure(tag,report) {
            if(window.dataActions.includes(tag))dataForm.applyErrors(report)
            if(window.modelActions.includes(tag) || tag==="preflight")modelForm.applyErrors(report)
            failureDialog.report=report;failureDialog.open()
        }
        function onResponse(tag,payload) {
            if(["devices","ports","checkpoint","preflight"].includes(tag)) {resultDialog.payload=payload;resultDialog.open()}
        }
    }
}
