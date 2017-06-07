function startTour(force_start){
    var upcoming_step = null; // hack to know upcoming step in onShow
    function next(t) {
        // console.log("next, cur: " + t.getCurrentStep());
        upcoming_step = t.getCurrentStep() + 1;
    }
    function prev(t) {
        // console.log("prev, cur: " + t.getCurrentStep());
        upcoming_step = t.getCurrentStep() - 1;
    }

    function autoScrollSideBar(t) {
        // sadly not set at this point
        // var element_offset = $("#sidebar-container .tour-step-backdrop").offset().top;

        // to be called onShow, hence upcoming_step hack
        var us = upcoming_step !== null ? upcoming_step : t.getCurrentStep();
        var element = t.getStep(us).element;
        if (!element) {
            console.warn("autoScrollSideBar called, but can't get element");
            return;
        }
        // var element_offset = $(element).offset().top;
        var panel = $(element).closest('.panel');
        $("#sidebar-container").scrollTo(panel, 200, {offset:-50});
        var panel_body = $(element).closest('.panel-body');
        if (panel_body) {
            panel_body.scrollTo(element, 200, {offset: -100});
            var tr = $(element).closest('tr');
            if (tr) {
                panel_body.scrollTo(tr, 200, {offset: -50});
            }
        }
    }

    function fixupSVGHighlighting(t) {
        // to be called onShown
        var element = t.getStep(t.getCurrentStep()).element;
        if (!element) {
            console.warn("fixupSVGHighlighting called, but can't get element");
            return;
        }

        // need to fix highlight area as it doesn't properly get dimensions
        var r = $(element)[0].getBoundingClientRect();
        // var p = tour.getOptions();
        // console.log($(element).width());
        // console.log($(element).height());
        $(".tour-step-background").width(r.width + 10).height(r.height + 10);

        // also raise the svg element above the backdrop
        $("#canvas").children("svg").attr("class", "tour-step-backdrop");
    }

    function fixupSVGHighlightingRemove(t) {
        $("#canvas").children("svg").removeAttr("class");
    }

    var tour = new Tour({
        backdrop: true,
        backdropPadding: 5,
        // autoscroll: true, // doesn't seem to work
        orphan: true,
        debug: true,
        onNext: next,
        onPrev: prev,
        steps: [
            {
                content: "Welcome to the result visualisation of our <a href='https://w3id.org/associations/#gp_learner'>graph pattern learner</a>.<br>As the interface is quite powerful, it can be overwhelming for newcomers. This tour will briefly walk you through all of its components."
            },
            {
                element: '#sidebar-global-info-panel',
                onShow: function (t) {
                    $("#graph-nav > a").click();
                    autoScrollSideBar(t)
                },
                placement: 'left',
                title: "Result step selection",
                content: "Let's start on the right. As you might know our evolutionary algorithm operates in several runs to cover the input source-target-pairs. Here you can select the overall aggregated results, the results of each run or even the results of an individual generation in a run."
            },
            {
                element: '#sidebar-select-panel',
                onShow: autoScrollSideBar,
                title: "Learned pattern selection",
                placement: 'left',
                content: "Once the selected results are loaded, you can select the individual patterns in this panel."
            },
            {
                element: '#sidebar-fitness-panel',
                onShow: autoScrollSideBar,
                title: "Pattern Fitness",
                placement: 'left',
                content: "In this panel you will see the fitness of the currently selected pattern.<br/>" +
                    "The most important dimensions of the fitness are:" +
                    "<ul>" +
                    "<li><b>score:</b> The overall score of this pattern.</li>" +
                    "<li><b>avg_reslens:</b> How noisy is this pattern? For a given ?source, how many ?targets does it return on average over the ground truth pairs?</li>" +
                    "<li><b>gt_matches:</b> How many of our ground truth source-target-pairs does this pattern actually cover? (shown in next step)</li>" +
                    "<li><b>qtime:</b> How long did it take to evaluate this pattern for all ground truth pairs?</li>" +
                    "</ul>"
            },
            {
                element: '#sidebar-pairs-panel',
                onShow: autoScrollSideBar,
                title: "Ground Truth Matches for pattern",
                placement: 'left',
                content: "Here you can see which of the ground truth source-target-pairs are matched by the selected pattern."
            },
            {
                element: '#sidebar-pairs-panel .pair-link:first',
                onShow: autoScrollSideBar,
                placement: 'left',
                content: "You can also execute the SPARQL query for the current pattern pre-filled with the individual source node."
            },
            {
                element: '#sidebar-SPARQL-panel',
                onShow: autoScrollSideBar,
                title: "Pattern's SPARQL query",
                placement: 'left',
                content: "Here you can find the SPARQL SELECT query for the selected pattern that was learned and can execute it."
            },
            {
                element: '#graph svg>g',
                onShown: fixupSVGHighlighting,
                onHide: fixupSVGHighlightingRemove,
                title: "Pattern Visualisation",
                content: "Instead of the (ugly) text version on the right, here you see a graphical representation of the learned and selected SPARQL pattern."
            },
            {
                element: '#graph svg>g',
                onShown: fixupSVGHighlighting,
                onHide: fixupSVGHighlightingRemove,
                title: "Pattern Visualisation",
                content: "The <span style='color:green'>?source</span> and <span style='color:red'>?target</span> are special variables that were filled with the ground truth pairs listed on the right during training."
            },
            {
                element: '#sidebar-select-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                backdrop: false,
                reflex: true,
                content: "Select another pattern and see how everything else is updated... (you may click!)"
            },
            {
                element: '#matrix-nav',
                onShow: autoScrollSideBar,
                placement: 'bottom',
                backdrop: false,
                reflex: true,
                onNext: function (t) {$("#matrix-nav > a").click(); next(t);},
                content: "After we've now seen how you can quickly explore the learned graph patterns, let's switch to the matrix tab. (click above)"
            },
            {
                element: '#main-content',
                // onShow: autoScrollSideBar,
                //placement: 'bottom',
                onPrev: function (t) {$("#graph-nav > a").click(); prev(t);},
                content: "The matrix tab gives an overview about how the patterns cover the ground truth pairs. Each block represents one ground truth pair. Just hover over them to see how the currently selected graph pattern covers each of them."
            },
            {
                element: '#matrix-container .matrix-div[data-original-title*="Precision: 1"]:first',
                // onShow: autoScrollSideBar,
                placement: 'auto bottom',
                backdrop: false,
                content: "For example, this is a high precision match of the current pattern selected on the right for the input ground truth pair (see the URIs on top). We also list how many patterns in total (including the current one) match it."
            },
            {
                element: '#matrix-container .matrix-div[data-original-title*="Precision: 0<"]:first',
                // onShow: autoScrollSideBar,
                placement: 'auto bottom',
                backdrop: false,
                content: "This is a low precision example. The currently selected pattern on the right does not match this ground truth pair (but other patterns hopefully do)."
            },
            {
                element: '#graph-radios input:not(:checked):first',
                onShow: autoScrollSideBar,
                placement: 'left',
                backdrop: false,
                reflex: true,
                content: "Selecting other patterns on the right will automatically update the matrix view."
            },
            {
                element: '#no-graph-radio-container',
                onShow: autoScrollSideBar,
                placement: 'left',
                backdrop: false,
                reflex: true,
                title: "Accumulated precisions",
                content: "In the matrix view you can also select this option to get an accumulated view that shows how <b>all patterns</b> in the current run and generation together cover the ground truth pairs."
            },
            {
                element: '#fingerprint-nav',
                // onShow: autoScrollSideBar,
                placement: 'bottom',
                backdrop: false,
                reflex: true,
                onNext: function (t) {$("#fingerprint-nav > a").click(); next(t);},
                content: "Let's now switch to the fingerprint view. (click above)"
            },
            {
                element: '#main-content',
                // onShow: autoScrollSideBar,
                //placement: 'bottom',
                onPrev: function (t) {$("#matrix-nav > a").click(); prev(t);},
                content: "Each line in the fingerprint view represents a learned pattern and shows its condensed matrix view. You can use these fingerprints to quickly find \"different\" patterns with respect to the ground truth."
            },
            {
                element: '#graph-fingerprint0',
                onShow: autoScrollSideBar,
                placement: 'bottom',
                backdrop: false,
                reflex: true,
                content: "As you might have guessed by now, you also see these fingerprints (only even more condensed) next to each of the graph patterns."
            },
            {
                content: "That's it. Feel free to explore!"
            }
        ]
    });
    tour.init();
    if (force_start) {
        tour.restart(0);
    }
    tour.start(force_start);
}
