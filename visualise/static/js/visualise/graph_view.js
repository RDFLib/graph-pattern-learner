/**
 * Requires d3 module to be loaded.
 */

GraphView = "GraphView" in window ? GraphView : {};


var FONTSIZE = 12;
var SHADOWSIZE = 3;
var INIT_LINK_WIDTH = 1;

GraphView.width = function() { return $('#canvas').width(); };
GraphView.height = function() { return $('#canvas').height(); };
GraphView.nodes = [];
GraphView.links = [];

GraphView.init = function() {

    $("#canvas").html(""); // clear canvas

    GraphView.forceLayout = d3.layout.force()
        .nodes(GraphView.nodes)
        .links(GraphView.links)
        .size([GraphView.width(), GraphView.height()])
        .linkDistance(200)
        .on("tick", GraphView.tick);

    var zoom = d3.behavior.zoom()
        .scaleExtent([1, 10])
        .on("zoom", GraphView.zoomed);

    var drag = d3.behavior.drag()
        .origin(function (d) { return d; })
        .on("dragstart", GraphView.dragStarted)
        .on("drag", GraphView.dragged)
        .on("dragend", GraphView.dragEnded);

    var svg = d3.select("#canvas").append("svg:svg")
        .attr("width", "100%")
        .attr("height", "100%")
        .call(zoom);

    GraphView.container = svg.append("g");

    // Per-type markers, as they don't inherit styles.
    GraphView.container.append("svg:defs").append("svg:marker")
        .attr("id", "marker")
        .attr("viewBox", "0 -5 10 10")
        .attr("refX", 20)
        .attr("refY", 0)
        .attr("markerWidth", 6)
        .attr("markerHeight", 6)
        .attr("orient", "auto")
        .attr("markerUnits","userSpaceOnUse")
        .append("svg:path")
        .attr("d", "M0,-5L10,0L0,5");

    GraphView.groups = {};

    GraphView.groups.link = GraphView.container.append("svg:g").attr("class", "links");
    GraphView.groups.linkDefs = GraphView.groups.link.append("svg:defs");
    GraphView.groups.linkPaths = GraphView.groups.link.append("svg:g").attr("class", "paths");
    GraphView.groups.linkLabels = GraphView.groups.link.append("svg:g").attr("class", "labels");
    GraphView.groups.linkLabelsShadow = GraphView.groups.linkLabels.append("svg:g").attr("class", "shadows");
    GraphView.groups.linkLabelsText = GraphView.groups.linkLabels.append("svg:g").attr("class", "texts");
    GraphView.groups.circles = GraphView.container.append("svg:g");
    GraphView.groups.text = GraphView.container.append("svg:g");
};


GraphView.start = function() {

    function getLinkID(data) {
        return btoa(encodeURIComponent(data.source.id)) + "_" +
            btoa(encodeURIComponent(data.target.id)) + "_" + data.count;
    }

    GraphView.joins = {};

    GraphView.joins.linkDefs = GraphView.groups.linkDefs.selectAll("path").data(GraphView.forceLayout.links());
    GraphView.joins.linkDefs.enter().append("path").attr('class', 'link-text-path');
    GraphView.joins.linkDefs.attr("id", getLinkID);
    GraphView.joins.linkDefs.exit().remove();

    GraphView.joins.link = GraphView.groups.linkPaths.selectAll(".links > .paths > path").data(GraphView.forceLayout.links());
    var link = GraphView.joins.link.enter().append('path').attr('class', 'link').style("stroke-width", INIT_LINK_WIDTH)
        .attr("marker-end", function() { return "url(#marker)"; });
    GraphView.joins.link.exit().remove();

    GraphView.joins.linkLabelsShadow = GraphView.groups.linkLabelsShadow.selectAll("text").data(GraphView.forceLayout.links());
    GraphView.joins.linkLabelsShadow
        .enter()
        .append("text")
        .append("textPath")
        .attr("class", "link-label-shadow-path")
        .attr("startOffset", "50%")
        .attr("text-anchor", "middle")
        .style("stroke", "#fff")
        .style("stroke-opacity", 0.8)
        .style("font-family", "Arial")
        .style("stroke-width", SHADOWSIZE)
        .style("font-size", FONTSIZE)
        .append("svg:tspan")
        .attr("class", "link-label-shadow")
        .attr("dy", "-2");
    GraphView.joins.linkLabelsShadow
        .select('.link-label-shadow-path')
        .attr("xlink:href", function(d) {
            return "#" + getLinkID(d);
        });
    GraphView.joins.linkLabelsShadow
        .select('.link-label-shadow')
        .text(function(d) { return d.label; });
    GraphView.joins.linkLabelsShadow.exit().remove();

    GraphView.joins.linkLabelsText = GraphView.groups.linkLabelsText.selectAll("text").data(GraphView.forceLayout.links());
    GraphView.joins.linkLabelsText
        .enter()
        .append("text")
        .append("textPath")
        .attr('class', 'link-label-text-path')
        .attr("startOffset", "50%")
        .attr("text-anchor", "middle")
        .style("fill", "#000")
        .style("font-family", "Arial")
        .style("font-size", FONTSIZE)
        .append("svg:tspan")
        .attr("class", "link-label-text")
        .attr("dy", "-2");
    GraphView.joins.linkLabelsText
        .select('.link-label-text-path')
        .attr("xlink:href", function(d) {
            return "#" + getLinkID(d)
        });
    GraphView.joins.linkLabelsText
        .select('.link-label-text')
        .text(function(d) { return d.label; });
    GraphView.joins.linkLabelsText.exit().remove();

    GraphView.joins.circles = GraphView.groups.circles.selectAll("circle")
        .data(GraphView.forceLayout.nodes(), function(d) {
            return d.id;
        });
    GraphView.joins.circles
        .enter()
        .append("svg:circle")
        .attr("r", 6)
        .attr("class", function(d) {
            var r = "";
            if (d.start) {
                r += "start-node ";
            }
            if (d.end) {
                r += "end-node";
            }
            return r;
        })
        .call(GraphView.forceLayout.drag)
        .on("mousedown", function() { d3.event.stopPropagation(); });
    GraphView.joins.circles.exit().remove();

    GraphView.joins.text = GraphView.groups.text.selectAll("g").data(GraphView.forceLayout.nodes());

    var text = GraphView.joins.text.enter().append("svg:g");
    GraphView.joins.text.exit().remove();

    // A copy of the text with a thick white stroke for legibility.
    text.append("svg:text")
        .attr("x", 8)
        .attr("y", ".31em")
        .attr("class", "shadow")
        .style("font-size", FONTSIZE)
        .style("stroke-width", SHADOWSIZE);

    text.append("svg:text")
        .attr("x", 8)
        .attr("y", ".31em")
        .attr("class", "node_label")
        .style("font-size", FONTSIZE);

    var howToText = GraphView.joins.text.selectAll("text").data(function(d) {return [d,d];});
    howToText.text(function(d) {return d.label;});

    GraphView.zoomed();

    $("#settings-scale-text, #settings-scale-arrow-tips").on('switchChange.bootstrapSwitch', function (e, state) {
        GraphView.zoomed();
    });

    $(GraphView.nodes).each(function(idx, node) {
        if (node.x === undefined) {
            node.x = Math.floor(Math.random() * GraphView.width());
            node.y = Math.floor(Math.random() * GraphView.height());
        }
    });

    GraphView.forceLayout.charge(-2000).start();

    GraphView.fixOrReleaseNodes(); // fixes all node positions if layout is paused
};

GraphView.resize = function(size) {
    GraphView.forceLayout.size(size);
    GraphView.start();
};

GraphView.fixOrReleaseNodes = function() {
    GraphView.joins.circles.each(function(d) {
        d.fixed = GraphView.paused;
    });
};

GraphView.resume = function() {
    GraphView.paused = false;
    GraphView.fixOrReleaseNodes();
};

GraphView.pause = function() {
    GraphView.paused = true;
    GraphView.fixOrReleaseNodes();
};

// Use elliptical arc path segments to doubly-encode directionality.
GraphView.tick = function() {
    function arcPath(leftHand, d) {
        var start = leftHand ? d.source : d.target;
        var c = d.count;
        var end = leftHand ? d.target : d.source;
        var dx = end.x - start.x;
        var dy = end.y - start.y;
        var sweep = leftHand ? 1 : 0;

        if ( dx != 0 || dy != 0 ) {
            sweep = (sweep+d.inverse)%2;
            if ((c - 1) % 2) {
                sweep = (!sweep)*1;
            }
            var h = Math.floor(c/2)*20;

            var norm = Math.sqrt(Math.pow(dx,2) + Math.pow(dy,2));
            var vcx, vcy;
            if (! sweep) {
                vcx = dy/norm;
                vcy = -dx/norm;
            } else {
                vcx = -dy/norm;
                vcy = dx/norm;
            }
            var cx = dx/2+vcx*h;
            var cy = dy/2+vcy*h;

            return "M" + start.x + "," + start.y + "q" + cx + ","
                + cy + " " + dx + "," + dy;
        } else {
            // self edge
            // Fiddle with this angle to get loop oriented.
            var xRotation = 90;
            // Needs to be 1.
            var largeArc = 1;
            // Change sweep to change orientation of loop.
            sweep = 1;
            // Make drx and dry different to get an ellipse
            // instead of a circle.
            var drx = 20;
            var dry = 30;
            var x1 = start.x;
            var y1 = start.y;
            // For whatever reason the arc collapses to a point if the beginning
            // and ending points of the arc are the same, so kludge it.
            var x2 = end.x + 1;
            var y2 = end.y + 1;
            return "M" + x1 + "," + y1 + "A" + drx + ","
                + dry + " " + xRotation + "," + largeArc + ","
                + sweep + " " + x2 + "," + y2;
        }
    }

    d3.selectAll(".link").attr("d", function(d) {
        return arcPath(true, d);
    });

    d3.selectAll(".link-text-path").attr("d", function(d) {
        return arcPath(d.source.x < d.target.x, d);
    });

    GraphView.joins.circles.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });

    GraphView.joins.text.attr("transform", function(d) {
        return "translate(" + d.x + "," + d.y + ")";
    });
};

GraphView.lastTranslate = null;
GraphView.lastScale = null;

GraphView.zoomed = function() {
    if (d3.event != null) {
        GraphView.lastTranslate = d3.event.translate;
        GraphView.lastScale     = d3.event.scale;
    }

    if (GraphView.lastScale == null || GraphView.lastTranslate == null) {
        return;
    }

    var translate = GraphView.lastTranslate;
    var scale     = GraphView.lastScale;

    var scaleText = $('#settings-scale-text').bootstrapSwitch('state');
    var scaleMarker = $('#settings-scale-arrow-tips').bootstrapSwitch('state');

    GraphView.container.attr("transform", "translate(" + translate + ")scale(" +
        scale + ")");

    var textScale = scaleText ? 1 : scale;
    GraphView.joins.text.selectAll("text").style("font-size", (FONTSIZE / textScale) + "px");
    GraphView.joins.text.selectAll(".shadow").style("stroke-width",
        SHADOWSIZE / textScale + "px");
    d3.selectAll(".link-label-shadow")
        .style("stroke-width", SHADOWSIZE / textScale + "px")
        .style("font-size", FONTSIZE / textScale + "px");
    d3.selectAll(".link-label-text").style("font-size",
        FONTSIZE / textScale + "px");

    var markerScale = scaleMarker ? 1 : scale;
    d3.select("#marker").attr("markerWidth", 5 / markerScale);
    d3.select("#marker").attr("markerHeight", 5 / markerScale);
    d3.select("#marker").attr("refX", 13.77777778 * markerScale + 6.222222222);

    d3.selectAll(".link").style("stroke-width", (INIT_LINK_WIDTH / scale) + "px")
};


GraphView.paused = false;

GraphView.dragStarted = function() {
    d3.select(this).classed("dragging", true);
};

GraphView.dragged = function(d) {
    d3.select(this).attr("cx", d.x = d3.event.x).attr("cy", d.y = d3.event.y);
};

GraphView.dragEnded = function() {
    d3.select(this).classed("dragging", false);
};


GraphView.updatePattern = function(pattern) {
    function calcSourceAndTarget(nodes, links) {
        var nodeDict = {};
        for (var i = 0; i < nodes.length; i++) {
            nodeDict[nodes[i]["id"]] = nodes[i];
        }
        for (var j = 0; j < links.length; j++) {
            var link = links[j];
            link["source"] = nodeDict[link["from"]];
            link["target"] = nodeDict[link["to"]];
        }
    }

    var newLinks = pattern["links"];
    var newNodes = pattern["nodes"];
    var newLinksDict = {};
    var newNodesDict = {};
    var i;
    for (i = 0; i < newLinks.length; i++) {
        newLinksDict[newLinks[i]["id"]] = i;
    }
    for (i = 0; i < newNodes.length; i++) {
        newNodesDict[newNodes[i]["id"]] = i;
    }

    for (i = 0; i < GraphView.links.length; i++) {
        // link is to be deleted
        GraphView.links.splice(i,1); i--;
    }
    // now add new links
    for (var key in newLinksDict) {
        GraphView.links.push(newLinks[newLinksDict[key]]);
    }

    for (i = 0; i < GraphView.nodes.length; i++) {
        if (typeof newNodesDict[GraphView.nodes[i]["id"]] == 'undefined') {
            // node is to be deleted
            GraphView.nodes.splice(i,1); i--;
        } else {
            // node exists => throw out of newNodesDict (keeping node as is)
            delete newNodesDict[GraphView.nodes[i]["id"]];
        }
    }
    // now add new nodes
    for (key in newNodesDict) {
        GraphView.nodes.push(newNodes[newNodesDict[key]]);
    }
    calcSourceAndTarget(GraphView.nodes, GraphView.links);
};


GraphView.reset = function() {
    $("#canvas").html("");
    if ("forceLayout" in GraphView) {
        GraphView.forceLayout.stop();
    }
    delete GraphView.forceLayout;
    delete GraphView.groups;
    delete GraphView.joins;
    GraphView.nodes = [];
    GraphView.links = [];
};


if (! ('d3' in window &&
       '$' in window &&
       'bootstrapSwitch' in $.fn)) {
    console.error("GraphView module needs d3, jquery, bootstrap, bootstrap-switch libs to be loaded.");
    delete GraphView;
}
