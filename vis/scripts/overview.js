var current_model_id = "simple_rnn"
var current_input_id = "childsplay"
var current_input = "The movie updates a classic horror icon for the Internet of Things era, with predictably gruesome, and generally entertaining, results."
var acts = null
var mode = "complete"
var zoom = "layer"
var ASSETS_PATH = "assets"

setActs()

function buildJSONFilename(model_id, input_id) {
    return ASSETS_PATH + "/" + model_id + "_" + input_id + ".json"
}

function setActs() {
    $.ajax({
        url: buildJSONFilename(current_model_id, current_input_id),
        async: false,
        dataType: 'json',
        success: function (response) {
          acts = response
        }
      });    
}

function setModel(new_id) {
    current_model_id = new_id
    setActs()
}

function setInput(new_id) {
    current_input_id = new_id
    setActs()
    
}

function selectActs(act, layer, t) {
    return acts[act+t+layer]
}
