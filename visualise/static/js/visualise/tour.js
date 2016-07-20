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
        $("#sidebar-container").scrollTo(element, 200, {offset:-50});
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
                content: "Welcome to the result visualisation of our <a href='https://w3id.org/associations/#gp_learner'>graph pattern learner</a>.<br>As the interface is quite powerful, this tour will briefly explain its components."
            },
            {
                element: '#sidebar-global-info-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                title: "Result step selection",
                content: "Let's start on the right. As you might know our evolutionary algorithm operates in several runs to cover the input source-target-pairs. Here you can select individual generations from each run, the results of each run or the overall aggregated results."
            },
            {
                element: '#sidebar-select-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                content: "Once the selected results are loaded, you can select the individual patterns in this panel."
            },
            {
                element: '#sidebar-fitness-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                content: "In this panel you will see the fitness of the currently selected pattern."
            },
            {
                element: '#sidebar-pairs-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                content: "Here you can see which of the ground truth source-target-pairs are matched by the selected pattern."
            },
            {
                element: '#sidebar-pairs-panel .pair-link:first',
                placement: 'left',
                content: "You can also execute the SPARQL query for the current pattern pre-filled with the individual source node."
            },
            {
                element: '#sidebar-SPARQL-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                content: "Here you can find the SPARQL SELECT query for the selected pattern that was learned and can execute it."
            },
            {
                element: '#graph svg>g',
                onShown: fixupSVGHighlighting,
                onHide: fixupSVGHighlightingRemove,
                content: "Instead of reading the ugly text version on the right, here you see a graphical representation of the pattern."
            },
            {
                element: '#graph svg>g',
                onShown: fixupSVGHighlighting,
                onHide: fixupSVGHighlightingRemove,
                content: "The ?source and ?target are special variables that were filled with the ground truth pairs on listed on the right during training."
            },
            {
                element: '#sidebar-select-panel',
                onShow: autoScrollSideBar,
                placement: 'left',
                backdrop: false,
                reflex: true,
                content: "Select another pattern and see how everything else is updated..."
            },
            {
                element: '#matrix-nav',
                onShow: autoScrollSideBar,
                placement: 'bottom',
                backdrop: false,
                reflex: true,
                onNext: function (t) {$("#matrix-nav > a").click(); next(t);},
                content: "After we've now seen how you can quickly explore the learned graph patterns, let's switch to the matrix tab."
            },
            {
                element: '#main-content',
                // onShow: autoScrollSideBar,
                //placement: 'bottom',
                onPrev: function (t) {$("#graph-nav > a").click(); prev(t);},
                content: "The matrix tab gives an overview about how the patterns cover the ground truth pairs. Each block represents one ground truth pair. Just hover over them to see how the currently selected graph pattern covers each of them."
            },
            {
                element: document.querySelectorAll('#step2')[0],
                content: "Ok, wasn't that fun?",
                position: 'right'
            },
            {
                element: '#step3',
                content: 'More features, more fun.',
                position: 'left'
            },
            {
                element: '#step4',
                content: "Another step.",
                position: 'bottom'
            },
            {
                element: '#step5',
                content: 'Get it, use it.'
            }
        ]
    });
    tour.init();
    if (force_start) {
        tour.restart(0);
    }
    tour.start(force_start);
}
