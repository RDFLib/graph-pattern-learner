/**
 * Requires d3 module to be loaded.
 */

MatrixView = "MatrixView" in window ? MatrixView : {};
MatrixView.Filter = "Filter" in MatrixView ? MatrixView.Filter : {};

MatrixView.matchesCB = null;
MatrixView.coverageMaxPrecision = null;


MatrixView.sort = function(byPrec) {
    if (byPrec === undefined || byPrec) {
        d3.selectAll(".matrix-div").sort(function(a, b) {
            // sort by precision
            var diff = b.value.prec - a.value.prec;
            if (diff === 0) {
                a = a.value.uri1 + a.value.uri2;
                b = b.value.uri1 + b.value.uri2;
                return a.localeCompare(b);
            }
            return diff;
        });
    } else {
        d3.selectAll(".matrix-div").sort(function (a, b) {
            // sort alphabetically
            a = a.value.uri1 + a.value.uri2;
            b = b.value.uri1 + b.value.uri2;
            return a.localeCompare(b);
        });
    }
};

MatrixView.update = function(coverageMaxPrecision, graphPatterns, graphIndex) {
    function escapeHTMLAll(array) {
        for (var i=0; i < array.length; i++) {
            array[i] = $("<div>").text(array[i]).html();
        }
        return array;
    }
    function nodesToId(nodes) {
        var res = [];
        for (var i=0; i < nodes.length; i++) {
            res.push(btoa(encodeURIComponent(nodes[i])));
        }
        return res.join("#");
    }

    MatrixView.coverageMaxPrecision = coverageMaxPrecision;
    MatrixView.graphs = graphPatterns;

    var cmp = {};
    var cmp_ = coverageMaxPrecision;
    var i;
    for (i=0; i < cmp_.length; i++) {
        var source = cmp_[i][0][0];
        var target = cmp_[i][0][1];
        // graphIndex === undefined means global accumulated precision
        var prec = graphIndex === undefined ? parseFloat(cmp_[i][1]) : 0;
        var matchCount = cmp_[i][2];
        cmp[nodesToId(cmp_[i][0])] = { "uri1": source,
            "uri2": target,
            "prec": prec,
            "matches": matchCount}
    }
    if (graphIndex !== undefined) {
        // showing specific graph pattern
        var pattern = graphPatterns[graphIndex];
        var gtps = pattern["gtp_precisions"];
        for (i=0; i < gtps.length; i++) {
            cmp[nodesToId(gtps[i][0])]["prec"] = gtps[i][1];
        }
    }

    var data = d3.entries(cmp);
    var matrixJoin = d3.select("#matrix-container").selectAll("div.matrix-div")
        .data(data, function(d) {
            return d.key;
        });
    matrixJoin.enter()
        .append("div")
        .attr("class", "matrix-div")
        .attr("id", function(d, i) {return "mat-elem-"+ i.toString()});
    matrixJoin.style("background-color", function(data) {
        var factor = data.value.prec;
        var rgb = Util.HSVtoRGB(206/180*3.1415,
            factor*.721,
            (1-factor)*.282+.718);
        return "rgb("+rgb.r + ", " + rgb.g + ", " + rgb.b + ")";
    })
        .attr('data-original-title', function(data) {
            // this attr belongs to bootstrap. Updating title attr does not
            // update the tooltip on data change
            data = data.value;
            var tooltip = [data.uri1, data.uri2, undefined, undefined];
            // truncate precision
            tooltip[2] = Math.round(parseFloat(data.prec)*1000)/1000;
            tooltip[2] = "Precision: " + tooltip[2].toString();
            tooltip[3] = "Matching patterns: " + data.matches;
            return escapeHTMLAll(tooltip).join("<br />");
        })
        .attr("onclick", function(d) {
            return "MatrixView.Filter.PatternsByMatches(\"" + d.value.uri1 + "\", \"" +
                d.value.uri2 + "\")";
        });

    matrixJoin.exit().remove();
    $(document).tooltip({
        placement: "auto bottom",
        html: "true",
        selector: ".matrix-div"
    });
    // sort matrix if button pressed
    MatrixView.sort($("#matrix-sort-btn").hasClass("active"));
};

MatrixView.init = function(coverageMaxPrecision, graphPatterns) {
    this.update(coverageMaxPrecision, graphPatterns);
};

MatrixView.reset = function() {
    $("#matrix-container").html("");
};


/**
 * Filter Methods
 */

MatrixView.Filter.clearPatterns = function() {
    $(".matrix-div").removeClass("matched");
};

MatrixView.Filter.PatternsByMatches = function(uri1, uri2, matchesCB) {
    // if only uri1 is given we filter by source and target
    // if both are given uri1 must match source and uri2 must match target
    // (both with infix search and case insensitive)

    // additionally uri1 is searched in the sparql query of each pattern

    var escapeRegEx = function(text) {
        return text.replace(/[-[\]{}()*+?.,\\^$|#\s]/g, "\\$&");
    };

    function matches(pair, uri1, uri2) {
        if (uri2 === undefined) {
            if (pair[0].toLowerCase().search(escapeRegEx(uri1.toLowerCase())) >= 0 ||
                pair[1].toLowerCase().search(escapeRegEx(uri1.toLowerCase())) >= 0) {
                // this is a match
                return true;
            }
        } else {
            if (pair[0].toLowerCase().search(escapeRegEx(uri1.toLowerCase())) >= 0 &&
                pair[1].toLowerCase().search(escapeRegEx(uri2.toLowerCase())) >= 0) {
                // this is a match
                return true;
            }
        }
        return false;
    }

    matchesCB = matchesCB || MatrixView.matchesCB;

    // handle searchbar
    $(".matrix-div").removeClass("matched");
    if (uri1 !== undefined && uri2 === undefined) {
        $("#searchbar").val(uri1);
    } else if (uri1 !== undefined && uri2 !== undefined) {
        $("#searchbar").val("");
    } else {
        console.error("Cannot search for ("+uri1+", "+uri2+")");
        return;
    }

    // highlight matched pairs in matrix view
    $(MatrixView.coverageMaxPrecision).each(function(idx, e){
        var pair = e[0];
        if (matches(pair, uri1, uri2)) {
            $("#mat-elem-"+idx.toString()).addClass("matched");
        }
    });

    // calculate matching graph patterns und hide other's radiobuttons
    var matchingPatternIdxs = [];
    $(MatrixView.graphs).each(function (idx, pattern) {
        $(pattern["matching_node_pairs"]).each(function (idx_, pair) {
            if (matches(pair, uri1, uri2)) {
                matchingPatternIdxs.push(idx);
                return false; // acts like break keyword
            }
        });

        // also search within the sparql query
        if (pattern["sparql_query"].toLowerCase().search(escapeRegEx(
                uri1.toLowerCase())) >= 0) {
            matchingPatternIdxs.push(idx);
        }
    });

    if(matchesCB) {
        // Sidebar.showHideRadios
        matchesCB(matchingPatternIdxs);
    }
};


if (! ('d3' in window)) {
    console.error("MatrixView module needs d3 lib to be loaded.");
    delete MatrixView;
}
