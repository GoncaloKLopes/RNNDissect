var REVIEWS_FILE = "assets/reviews.json"
var input_flag = false
var text_flag = false
var aux_text = ""
var available_input_selected = null
var available_model_selected = null


$(document).ready(function() {
    //SET INPUT MODAL
    $("#set-input").children("span").click(function() {

        if (!input_flag){
            input_flag = true
            $.getJSON(REVIEWS_FILE, function(json) {
                $.each(json, function(key, val) {
                    $("#available-inputs").append("<li id=" + val.id + ">" + val.text +"</li>")
                    if (val.id == current_input_id) {
                        $("#" + val.id).addClass("available-input-selected")
                        available_input_selected = $("#"+ val.id)
                        $("#set-input-save").removeClass("inactive")
                    }
                })
            })
        }
        $("#set-input-modal").modal({
            backdrop: "static",
            keyboard: false
        })
    })

    $("#set-input-text").on("keyup", function() {
        if($(this).val().length > 0){
            $("#set-input-save").removeClass("inactive")
            $(available_input_selected).removeClass("available-input-selected")
        }
        else {
            $("#set-input-save").addClass("inactive")
        }
    })
    $("#available-inputs").on("click", "li", function() {
        $("#set-input-text").val("")
        if (available_input_selected != null || available_input_selected == this) {
            $(available_input_selected).removeClass("available-input-selected")
        }
        if(available_input_selected != this) {
            $(this).addClass("available-input-selected")
            available_input_selected = $(this)
            $("#set-input-save").removeClass("inactive")
        }
        else {
            available_input_selected = null
            $("#set-input-save").addClass("inactive")
        } 
    })

    $("#set-input-save").on("click", function() {
        if (!$(this).hasClass("inactive")){
            if($("#set-input-text").val() != "") {
                current_input_id = null
                current_input = $("#set-input-text").val()
                text_flag = true
                aux_text = current_input
            }
            else {
                setInput(available_input_selected.attr("id"))
                current_input = available_input_selected.text()
                text_flag = false
                
            }
            //TODO overview stuff
            $("#set-input-modal").modal("hide")
        }
    })

    $("#set-input-cancel").on("click", function() {
        if (!text_flag) {
            $("#set-input-text").val("")
            available_input_selected.removeClass("available-input-selected")
            available_input_selected = null
            
            $("#" + current_input_id).addClass("available-input-selected")
            available_input_selected = $("#" + current_input_id)
            $("#set-input-save").removeClass("inactive")
        }
        else {
            $("#set-input-text").val(current_input)
        }
        
        $("#set-input-modal").modal("hide")
    })

    //SET MODEL MODAL
    $("#set-model").children("span").click(function() {

        $("#available-models > #" + current_model_id).addClass("available-model-selected")
        available_model_selected = $("#available-models > #" + current_model_id)
        $("#set-model-modal").modal({
            backdrop: "static",
            keyboard: false
        })
    })

    $("#available-models").on("click", "li", function() {
        if (available_model_selected != null || available_model_selected == this) {
            $(available_model_selected).removeClass("available-model-selected")
        }
        if(available_model_selected != this) {
            $(this).addClass("available-model-selected")
            available_model_selected = $(this)
            $("#set-model-save").removeClass("inactive")
        }
        else {
            available_model_selected = null
            $("#set-model-save").addClass("inactive")
        } 
    })

    $("#set-model-save").on("click", function() {
        setModel(available_model_selected.attr("id"))
        //TODO overview stuff
        $("#set-model-modal").modal("hide")
    })

    $("#set-model-cancel").on("click", function() {
        $(available_model_selected).removeClass("available-model-selected")

        $("#available-models > #" + current_model_id).addClass("available-model-selected")
        available_model_selected = $("#available-models > #" + current_model_id)
        $("#set-model-modal").modal("hide")
    })

    //OTHER SETTINGS
    $("#other-settings").children("span").click(function() {
        $("#other-settings-mode-" + mode).prop("checked", true)
        $("#other-settings-acts-" + zoom).prop("checked", true)
        $("#settings-modal").modal({
            backdrop: "static",
            keyboard: false
        })
    })

    $("#settings-save").on("click", function() {
        mode = $("input[name=mode]:checked").val()
        zoom = $("input[name=acts]:checked").val()
        $("#settings-modal").modal("hide")
    })

    $("#settings-cancel").on("click", function() {
        $("#settings-modal").modal("hide")
    })

})

