/**
  * Module needs
 *  crowbar.js
 *  graph_view.js
 *  matrix_view.js
 *  sidebar.js
 *  data_loader.js
 *
 * to be loaded and global vars
 *  START_FILENAME
 *  RUNS_GENS_DICT
 *
 * to be set.
 */

Main = "Main" in window ? Main : {};

Main.curRun = null;
Main.curGen = null;

// maximum run as default
Main.lastRealRun = Object.keys(RUNS_GENS_DICT).reduce(function(a, b){ return parseInt(a) > parseInt(b) ? parseInt(a) : parseInt(b) });
// maximum gen of max run as default
Main.lastRealGen = RUNS_GENS_DICT[Main.lastRealRun];

Main.main = function() {

    function startLoadCB(fn) {
        document.title = "loading "+fn+" ...";
        console.log('loading ' + fn);
        Main.startWaitScreen();
    }
    function dataLoadDoneCB(data, fn) {
        Main.graphsParsed = data;
        document.title = fn+"#"+data.timestamp+"#NoGraph";
        Main.init();
    }
    function dataLoadFailCB() {
        Sidebar.updateRunGenFields(Main.lastRealRun, Main.lastRealGen);
    }
    function dataLoadAlwaysCB() {
        Main.stopWaitScreen();
    }

    function matrixFilterMatchCB(matchedIdxs) {
        Sidebar.showHideRadios(matchedIdxs);
        Sidebar.fallbackToFirstRadio();
    }

    $("#sidebar-container").sidebarize(function(event, ui) {
        $(".right").css("min-width", ui.size.width.toString() + "px");
        $("#canvas").resize();
    });
    Sidebar.radioCB = Main.showGraph;

    DataLoader.startCB = startLoadCB;
    DataLoader.resultCB = dataLoadDoneCB;
    DataLoader.failCB = dataLoadFailCB;
    DataLoader.alwaysCB = dataLoadAlwaysCB;

    MatrixView.matchesCB = matrixFilterMatchCB;

    var params = Util.getQueryParams();
    if ("fn" in params) {
        DataLoader.loadFile(params.fn);
    } else {
        DataLoader.loadFile(START_FILENAME);
    }
    Main.setBindings();
};

Main.init = function() {
    var accum = Sidebar.accumulatedRadioSelected();

    Main.resetAll();

    var gn = Main.graphsParsed["generation_number"];
    var rn = Main.graphsParsed["run_number"];
    if (gn != -1) Main.lastRealGen = gn;
    if (rn != -1) Main.lastRealRun = rn;


    Main.curGen = Main.graphsParsed["generation_number"];
    Main.curRun = Main.graphsParsed["run_number"];

    Sidebar.init(Main.graphsParsed, Main.curRun, Main.curGen);
    GraphView.init();
    MatrixView.init(
        Main.graphsParsed['coverage_max_precision'],
        Main.graphsParsed["graphs"]);
    FingerprintView.init(
        Main.graphsParsed['coverage_max_precision'],
        Main.graphsParsed["graphs"],
        Sidebar.clickGraphPattern.bind(Sidebar));

    Sidebar.clickFirstRadio(!(accum || Main.graphsParsed.graphs.length === 0));
    Sidebar.addFingerprints(FingerprintView); // has to happen after FingerprintView.init

};

Main.setBindings = function() {
    function runOrGenChanged() {
        var r = $("#info-run").val();
        var g = $("#info-generation").val();
        if (Main.curRun !== parseInt(r) && parseInt(g) >= 0) {
            // display new run, so start with last gen
            g = RUNS_GENS_DICT[r];
        }
        DataLoader.loadRunGen(r, g);
    }

    $("#info-generation").change(runOrGenChanged);
    $("#info-run").change(runOrGenChanged);

    $("#play-btn").click(function() {
        if (!$(this).hasClass("active")) {
            $("#play-pause-bar").find(".btn").removeClass("active");
            $(this).addClass("active");
            GraphView.resume();
        }
    });
    $("#pause-btn").click(function() {
        if (!$(this).hasClass("active")) {
            $("#play-pause-bar").find(".btn").removeClass("active");
            $(this).addClass("active");
            GraphView.pause();
        }
    });
    $("#download-btn").click(function() {
        crowbar();
        $(this).blur();  // remove focus from button
    });

    $("#matrix-sort-btn").click(function() {
        $(this).toggleClass("active");
        MatrixView.sort($(this).hasClass("active"));
        $(this).blur();  // remove focus from button
        return false;
    });

    $('input[type=radio][name=file-type]').change(function() {
        var r = Main.lastRealRun;
        var g = Main.lastRealGen;
        //noinspection FallThroughInSwitchStatementJS
        switch ($(this).val()) {
            case 'fin':
                r = -1;
            case 'run':
                g = -1;
        }
        DataLoader.loadRunGen(r, g);
    });

    $("#matrix-nav").find("> a").click(function() {
        $("#no-graph-radio-container").show();
        MatrixView.update(Main.graphsParsed['coverage_max_precision'],
            Main.graphsParsed["graphs"], Sidebar.getSelected());
    });

    var graph_nav = $("#graph-nav");
    graph_nav.find("> a").click(function() {
        $("#no-graph-radio-container").hide();
        Sidebar.fallbackToFirstRadio();
    });

    $('#fingerprint-nav > a').click(function() {
        $("#no-graph-radio-container").show();
        FingerprintView.selectGP(Sidebar.getSelected());
    });

    $(window).keydown(function(e) {
        if (e.keyCode === 70 && e.altKey && e.shiftKey && !e.metaKey && !e.ctrlKey) {
            $("#searchbar").focus();
            return false;
        }
    });

    $("#searchbar").bind('input', function() {
        var val = $(this).val();
        if (! val) {
            MatrixView.Filter.clearPatterns();
            Sidebar.showAllRadios();
        } else {
            MatrixView.Filter.PatternsByMatches($(this).val());
        }
    });

    $("#help-icon").click(function() {startTour(true)});
    $(function() {startTour(false)});

    $("input.switchify[type='checkbox']").bootstrapSwitch();


    $("#matrix").click(function(e) {
        if (!$(e.target).hasClass("matrix-div")){
            MatrixView.Filter.clearPatterns();
            Sidebar.showAllRadios();
            $("#searchbar").val("");
        }
    });

    $(window).resize(function() {
        $("#canvas").resize();
    });

    $("#canvas").resize(function() {
        var size = [$(this).width(), $(this).height()];
        if ($(this).is(":visible")) {
            GraphView.resize(size);
        }
        return false;
    });

    graph_nav.find("> a").on("shown.bs.tab", function() {
        $("#canvas").resize();
    });
};

Main.showGraph = function(graphIndex) {
    if (typeof(graphIndex) != 'undefined') {
        GraphView.updatePattern(Main.graphsParsed['graphs'][graphIndex]);

        if ($("#matrix-nav").hasClass("active") || MatrixView.graphs == null) {
            MatrixView.update(Main.graphsParsed['coverage_max_precision'],
                Main.graphsParsed['graphs'], graphIndex);
        }
        GraphView.start();

        Sidebar.updateGraphInfo(Main.graphsParsed['graphs'][graphIndex], graphIndex);
    } else {
        Sidebar.clearGraphInfo();
        MatrixView.update(Main.graphsParsed['coverage_max_precision'], Main.graphsParsed['graphs']);
    }
    FingerprintView.selectGP(graphIndex);

    var t = document.title.split("#");
    if (graphIndex != null) {
        document.title = t.slice(0, -1).join("#")+"#Graph_"+graphIndex.toString();
    } else {
        document.title = t.slice(0, -1).join("#");
    }
};

Main.resetAll = function() {
    // does NOT reset the play/pause bar
    Sidebar.reset();
    // reset canvas
    GraphView.reset();
    // reset matrix view
    MatrixView.reset();
    // reset fingerprint view
    FingerprintView.reset();
};

Main.startWaitScreen = function() {
    $(".modal-load-overlay").removeClass('collapsed');
};
Main.stopWaitScreen = function() {
    $(".modal-load-overlay").addClass('collapsed');
};


/**
 * Init
 */
if ('START_FILENAME' in window &&
    'RUNS_GENS_DICT' in window &&
    'crowbar' in window &&
    'GraphView' in window &&
    'MatrixView' in window &&
    'FingerprintView' in window &&
    'Sidebar' in window &&
    'DataLoader' in window) {

    $(document).ready(function() {
        Main.main();
    });
} else {
    console.error("Dependencies for main.js not fulfilled.\n Expecting:\n" +
        "crowbar.js, graph_view.js, matrix_view.js, FingerprintView.js, sidebar.js," +
        "data_loader.js modules and\nSTART_FILENAME, RUNS_GENS_DICT global vars.");
    delete Main;
}
