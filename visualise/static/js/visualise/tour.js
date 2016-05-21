function startTour(force_start){
    function autoScrollSideBar(t) {
        // to be called onShow, hence the currentstep + 1 below
        var element = t.getStep(t.getCurrentStep() + 1).element;
        if (!element) {
            console.warn("autoScrollSideBar called, but can't get element");
            return;
        }
        var element_offset = $(element).offset().top;
        $("#sidebar-container").animate({scrollTop: element_offset}, 200);
    }

    function fixupSVGHighlighting(t) {
        // to be called onShown
        var element = t.getStep(t.getCurrentStep()).element;
        if (!element) {
            console.warn("fixupSVGHighlighting called, but can't get element");
            return;
        }
        console.log('dim:');
        console.log($(element).width());
        console.log($(element).height());
        $(".tour-step-background").width($(element).width()).height($(element).height());
    }

    var tour = new Tour({
        backdrop: true,
        backdropPadding: 5,
        // autoscroll: true, // doesn't seem to work
        orphan: true,
        debug: true,
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
                element: '#graph svg>g',
                onShown: fixupSVGHighlighting,
                content: "This is a tooltip."
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
