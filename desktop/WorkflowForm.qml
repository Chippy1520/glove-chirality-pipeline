import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: form
    property var definition: ({fields: []})
    property var values: ({})
    property var errors: ({})
    property bool advancedOpen: false
    property bool actionsVisible: true
    property bool locked: false
    property string stopSlot: "pipeline"
    property string buttonLabel: "Start workflow"
    signal submitted(var payload)
    signal valuesEdited()
    spacing: 16

    function defaults() {
        const result = {}
        for (const field of (definition.fields || [])) result[field.name] = field.value === undefined ? "" : field.value
        values = result
        errors = {}
    }
    function setValue(name, value) {
        const next = Object.assign({}, values)
        next[name] = value
        values = next
        valuesEdited()
    }
    function applyErrors(report) {
        const result = {}
        const checks = report.preflight ? report.preflight.checks : report.checks
        if (Array.isArray(checks)) for (const check of checks) if (check.status === "failed") result[check.field] = check.message || "Invalid value"
        errors = result
        advancedOpen = true
    }
    onDefinitionChanged: defaults()
    Component.onCompleted: defaults()

    GridLayout {
        Layout.fillWidth: true
        columns: width < 650 ? 1 : 2
        columnSpacing: 20
        rowSpacing: 16
        Repeater {
            model: form.definition.fields || []
            delegate: ColumnLayout {
                id: fieldRow
                required property var modelData
                property var field: modelData
                Layout.fillWidth: true
                Layout.minimumWidth: 200
                visible: !field.advanced || form.advancedOpen
                spacing: 5
                Label {
                    visible: field.kind !== "checkbox"
                    text: field.title + (field.required ? " *" : "")
                    font.pixelSize: 13
                    font.weight: Font.DemiBold
                    color: "#344054"
                    wrapMode: Text.Wrap
                }
                Loader {
                    Layout.fillWidth: true
                    active: fieldRow.visible
                    enabled: !form.locked || ["mode","reject_class","decision_class","decision_threshold","delay_ms"].includes(field.name)
                    sourceComponent: field.kind === "checkbox" ? checkField : field.kind === "select" ? selectField : textField
                }
                Label {
                    visible: Boolean(form.errors[field.name])
                    text: form.errors[field.name] || ""
                    color: "#b42318"
                    font.pixelSize: 12
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }
                Label {
                    visible: Boolean(field.help) && (form.advancedOpen || Boolean(field.help_always))
                    text: field.help || ""
                    color: "#667085"
                    font.pixelSize: 12
                    wrapMode: Text.Wrap
                    Layout.fillWidth: true
                }
                Component {
                    id: checkField
                    CheckBox {
                        objectName: field.name
                        text: field.title
                        checked: Boolean(form.values[field.name])
                        onToggled: form.setValue(field.name, checked)
                    }
                }
                Component {
                    id: selectField
                    ComboBox {
                        objectName: field.name
                        model: field.choices || []
                        textRole: "label"
                        valueRole: "value"
                        currentIndex: {
                            const choices = field.choices || []
                            for (let i=0; i<choices.length; ++i) if (String(choices[i].value) === String(form.values[field.name])) return i
                            return 0
                        }
                        onActivated: form.setValue(field.name, currentValue)
                    }
                }
                Component {
                    id: textField
                    RowLayout {
                        TextField {
                            objectName: field.name
                            Layout.fillWidth: true
                            text: String(form.values[field.name] === undefined ? "" : form.values[field.name])
                            selectByMouse: true
                            placeholderText: field.required ? "Required" : "Use default"
                            inputMethodHints: field.kind === "number" ? Qt.ImhFormattedNumbersOnly : Qt.ImhNone
                            onEditingFinished: form.setValue(field.name, text)
                        }
                        Button {
                            visible: Boolean(field.browse)
                            text: "Browse"
                            onClicked: {
                                const chosen = client.choosePath(field.browse, String(form.values[field.name] || ""))
                                if (chosen) form.setValue(field.name, chosen)
                            }
                        }
                    }
                }
            }
        }
    }
    ToolButton {
        text: form.advancedOpen ? "Hide advanced settings" : "Advanced settings"
        visible: (form.definition.fields || []).some(field => field.advanced)
        onClicked: form.advancedOpen = !form.advancedOpen
    }
    RowLayout {
        visible: form.actionsVisible
        Button {
            text: form.buttonLabel
            highlighted: true
            enabled: client.ready
            onClicked: form.submitted(form.values)
        }
        Button {
            text: "Stop workflow"
            enabled: client.ready
            onClicked: client.request("stop", "/api/stop/" + form.stopSlot, "POST", {})
        }
    }
}
