/**
 * Requires global var RUNS_GENS_DICT to be set.
 */

Sidebar = "Sidebar" in window ? Sidebar : {};

Sidebar.radioCB = null;

$.fn.sidebarize = function(resizeCB) {
    var sidebar = this;
    var minWidth = sidebar.css("min-width");
    var index = minWidth.indexOf("px");
    if(index != -1)
        minWidth = minWidth.substr(0, index);
    this.resizable({
        handles: 'w',
        minWidth: minWidth,
        resize: function (event, ui) {
            sidebar.css("width", "100%");
            sidebar.css("left", "unset");
            if(resizeCB !== undefined) {
                resizeCB(event, ui);
            }
        }
    });
    return this
};

Sidebar.reset = function() {
    // reset side bar
    $("#info-run").val("");
    $("#info-generation").val("");
    $("#info-timestamp").html("");
    $("#info-permalink").attr("href", "");
    $("#graph-radios").html("");
    $("#info-fitness").html("<tr><td>No pattern found</td></tr>");
    $("#sparql-query").html("");
    $("#sparql-link").attr("href", "");
    $("#collapse-pairs-data").html("");
    $("#searchbar").val("");
};

Sidebar.init = function(data, curRun, curGen) {
    Sidebar.updateFileInfoAndHistory(curRun, curGen, curRun, curGen, data);
    $('#info-run').attr('max', Math.max.apply(null,
        $.map(Object.keys(RUNS_GENS_DICT), function(e) {
            return parseInt(e);
        })
    ));

    $('#no-graph-radio').attr('onclick', 'Sidebar.radioClicked()');
};

Sidebar.updateRunGenFields = function(lastRealRun, lastRealGen) {
    var runSel = $("#info-run");
    var genSel = $("#info-generation");
    var runAct = true;
    var genAct = true;
    //noinspection FallThroughInSwitchStatementJS
    switch ($('input[type=radio][name=file-type]:checked').val()) {
        case 'fin':
            runAct = false;
            lastRealRun = "";
        case 'run':
            genAct = false;
            lastRealGen = "";
            break;
    }
    runSel.val(lastRealRun);
    genSel.val(lastRealGen);
    runSel.prop('disabled', !runAct);
    genSel.prop('disabled', !genAct);
};

Sidebar.clearGraphInfo = function() {
    var colPairData = $("#collapse-pairs-data");
    var infoFitness = $("#info-fitness");
    infoFitness.html("");
    colPairData.html("");
    $("#sparql-link")
        .attr("href", "");
    $("#sparql-query")
        .text("");
};

Sidebar.updateGraphInfo = function(pattern, graphIndex) {
    // parsedGraph = Main.graphsParsed["graphs"][graphIndex]
    var infoFitness = $("#info-fitness");
    infoFitness.html("");
    var fitness = pattern["fitness"];
    fitness = [(graphIndex+1).toString()].concat(fitness);
    var fitnessDescription = ["Pattern number"].concat(
        pattern["fitness_description"]);
    if (fitness.length == fitnessDescription.length) {
        for (var i = 0; i < fitness.length; i++) {
            var tr = $("<tr>");
            tr.append($("<td>").text(fitnessDescription[i]));
            tr.append($("<td>").text(Math.round(fitness[i]*1000)/1000));
            infoFitness.append(tr);
        }
    } else {
        infoFitness.html("<tr><th>BROKEN DATA!</th></tr>");
    }
    var colPairData = $("#collapse-pairs-data");
    colPairData.html("");
    var matchingPairs = pattern["matching_node_pairs"];
    for (i = 0; i < matchingPairs.length; i++) {
        tr = $("<tr>");
        tr.append("<td class=\"pair-link\">" +
            '<a href="'+matchingPairs[i][2]+'" target="_blank">' +
            '<span class="glyphicon glyphicon-share" aria-hidden="true">' +
            "</span></a>" +
            "</td>");
        var td = $("<td></td>");
        var s = matchingPairs[i][0];
        var sLink = s.substring(1, s.length - 1);
        var t = matchingPairs[i][1];
        var tLink = t.substring(1, t.length - 1);
        var pairDesc = $('<dl class="dl-horizontal"></dl>');
        pairDesc.append('<dt>?source</dt>');
        var dd = $('<dd></dd>');
        dd.append($('<a href="' + sLink + '" target="_blank">').text(s));
        pairDesc.append(dd);
        pairDesc.append('<dt>?target</dt>');
        dd = $('<dd></dd>');
        dd.append($('<a href="' + tLink + '" target="_blank">').text(t));
        pairDesc.append(dd);
        td.append(pairDesc);
        tr.append(td);
        colPairData.append(tr);
    }

    $("#sparql-link")
        .attr("href", pattern["sparql_link"]);
    $("#sparql-query")
        .text(pattern["sparql_query"]);
};

Sidebar.updateFileInfoAndHistory = function ( currentRun, currentGeneration,
                                              lastRealRun, lastRealGeneration,
                                              graphsParsed ) {
    if (currentRun == -1 && currentGeneration == -1) {
        $("#ft-fin").prop("checked", true);
    } else if (currentRun != -1 && currentGeneration == -1) {
        $("#ft-fin-run").prop("checked", true);
    } else {
        $("#ft-gen").prop("checked", true);
    }
    Sidebar.updateRunGenFields(lastRealRun, lastRealGeneration);

    $("#info-timestamp").text(graphsParsed["timestamp"].split(".")[0]);
    var permLinkSuffix = "?fn="+graphsParsed["filename"];
    $("#info-permalink").attr("href", permLinkSuffix);
    window.history.pushState(
        permLinkSuffix, "", permLinkSuffix);

    var radioContainer = $("#graph-radios");
    radioContainer.html("");
    for (var i = 0; i < graphsParsed["graphs"].length; i++) {
        var rbId = "graph-radio" + i.toString();
        var r = $("<input>").attr("id",rbId)
            .attr("name", "graph-select")
            .attr("type", "radio")
            .attr("onclick", "Sidebar.radioClicked(" + i.toString() + ")")
            .attr("value", i.toString())
            .addClass("graph-radio");
        var l = $("<label>").attr("for", rbId)
            .html(" Show graph number "+ (i+1).toString())
            .addClass("graph-radio-label");
        radioContainer.append(l.prepend(r));
    }
    $('#info-generation').attr('max', RUNS_GENS_DICT[currentRun] || 0)
};

Sidebar.addFingerprints = function(FingerprintView) {
    $("#graph-radios").find('label').each(function() {
        var container = $('<div class="radio-fingerprint"></div>');
        $(this).prepend(container);
        var graphIndex = parseInt($(this).find('input').attr('value'));
        container.attr('id', 'graph-fingerprint'+graphIndex.toString());
        FingerprintView.addFingerPrint(container, graphIndex);
    });
    $("#no-graph-radio-container").find('label').each(function() {
        // make sure that old fingerprint is gone (e.g., previous run)
        $(this).find(".radio-fingerprint").remove();
        // add new fingerprint
        var container = $('<div class="radio-fingerprint"></div>');
        container.attr('id', 'no-graph-fingerprint');
        $(this).prepend(container);
        FingerprintView.addFingerPrint(container);
    })
};

Sidebar.radioClicked = function(idx) {

    // scroll selected radio button of pattern into view
    var container = $('#collapse-select');
    var scrollTo = [];
    if (idx != null) {
        scrollTo = $('#graph-radio' + idx);
    } else {
        scrollTo = $('#no-graph-radio');
    }
    if (scrollTo.length) {
        container.scrollTop(
            scrollTo.offset().top - container.offset().top + container.scrollTop() - 120
        );
    }

    if (Sidebar.radioCB) {
        Sidebar.radioCB(idx);
    }
};

Sidebar.showHideRadios = function(visible_idxs) {
    $("#graph-radios").find(".graph-radio-label").each(function(idx) {
        if (visible_idxs.indexOf(idx) >= 0) {
            $(this).show();
            $(this).children().first().attr('disabled', false);
        } else {
            $(this).hide();
            $(this).children().first().attr('disabled', true);
        }
    })
};

Sidebar.showAllRadios = function() {
    $("#graph-radios").find(".graph-radio-label").each(function() {
        $(this).show().children().first().attr('disabled', false);
    });
};

Sidebar.accumulatedRadioSelected = function() {
    return $("#no-graph-radio").is(":checked");
};

Sidebar.fallbackToFirstRadio = function() {

    if (!$("#no-graph-radio").is(":checked:visible")) {
        var checked = $(".graph-radio:checked").not("#no-graph-radio");
        var checkedIsDisplayed = checked.is(":visible");
        if ((checked.length === 0 || !checkedIsDisplayed)) {
            // select first radio instead
            Sidebar.clickFirstRadio(true);
        }
    }
};

Sidebar.clickFirstRadio = function(ignoreAccumulated) {
    if (typeof(ignoreAccumulated) == 'undefined') {
        ignoreAccumulated = !Sidebar.accumulatedRadioSelected();
    }
    if (ignoreAccumulated) {
        $(".graph-radio:visible").not("#no-graph-radio").first().click();
    } else {
        $(".graph-radio:visible:first").click();
    }
};

Sidebar.clickGraphPattern = function(graphIndex) {
    if (graphIndex == null) $(".graph-radio#no-graph-radio").click();
    else $("#graph-radio"+graphIndex.toString()).click();
};

Sidebar.getSelected = function() {
    var selected = $(".graph-radio:checked").val();
    if (selected === undefined || parseInt(selected) < 0) {
        selected = undefined;
    } else {
        selected = parseInt(selected);
    }
    return selected;
};


if (!('RUNS_GENS_DICT' in window)) {
    console.error("sidebar.js requires global var RUNS_GENS_DICT to be set.");
    delete Sidebar;
}
