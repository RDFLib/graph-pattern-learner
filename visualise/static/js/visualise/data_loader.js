/**
 * Expect global var SCRIPT_ROOT to be set and module Zlib to be loaded.
 */

DataLoader = "DataLoader" in window ? DataLoader : {};

DataLoader.startCB = null;
DataLoader.resultCB = null;
DataLoader.failCB = null;
DataLoader.alwaysCB = null;


DataLoader.decompressData = function(data) {
    // inspired here: http://dougbtv.com/2014/04/16/decompressing-gzipped-with-javascript/
    // Create a byte array.
    var bytes = [];

    // Walk through each character in the stream.
    for (var fileidx = 0; fileidx < data.length; fileidx++) {
        var abyte = data.charCodeAt(fileidx) & 0xff;
        bytes.push(abyte);
    }

    var gunzip = new Zlib.Gunzip(bytes);
    var plain = gunzip.decompress();

    var asciistring = "";
    for (var i = 0; i < plain.length; i++) {
         asciistring += String.fromCharCode(plain[i]);
    }

    return asciistring;
};


DataLoader.loadFile = function(fn, startCB, resultCB, failCB, alwaysCB) {
    startCB = startCB || DataLoader.startCB;
    resultCB = resultCB || DataLoader.resultCB;
    failCB = failCB || DataLoader.failCB;
    alwaysCB = alwaysCB || DataLoader.alwaysCB;
    var fnLoc = SCRIPT_ROOT + '/' + fn;
    if (startCB) {
        startCB(fn);
    }
    return $.ajax(fnLoc, { beforeSend: function( xhr ) {
        xhr.overrideMimeType( "text/plain; charset=x-user-defined" );
    }}).success(function(data) {
        console.log('ajax success, extracting...');
        data = DataLoader.decompressData(data);
        console.log('extracting done, loading...');
        data = DataLoader.prepareData($.parseJSON(data));
        if (resultCB) {
            resultCB(data, fn);
        }
        console.log('done.');
    }).fail(function(e) {
        console.error(e);
        if (failCB) {
            failCB(e, fn);
        }
    }).always(function() {
        console.log('ajax always');
        if (alwaysCB) {
            alwaysCB(fn);
        }
    });
};

DataLoader.prepareData = function(data) {

    function calcMatchingPatternsCount(coverageMaxPrec, allPatterns) {
        for (var i=0; i < coverageMaxPrec.length; i++) {
            var pair1 = coverageMaxPrec[i][0];
            var matchCount = 0;
            for (var j = 0; j < allPatterns.length; j++) {
                var pattern = allPatterns[j];
                for (var k = 0; k < pattern["matching_node_pairs"].length; k++) {
                    var pair2 = pattern["matching_node_pairs"][k].slice(0, 2);
                    if (pair1[0] === pair2[0] && pair1[1] === pair2[1]) {
                        matchCount++;
                        break;
                    }
                }
            }
            coverageMaxPrec[i].push(matchCount);
        }
        return coverageMaxPrec;
    }

    function calcLinkCount(graphList) {
        for (var i = 0; i < graphList.length; i++) {
            var graph = graphList[i];
            var set = {};
            var links = graph["links"];
            var nodes = graph["nodes"];
            for (var j = 0; j < links.length; j++) {
                var link = links[j];
                var source = link["from"];
                var target = link["to"];
                var key1 = [source, target];
                var key2 = [target, source];
                if (key1 in set) {
                    set[key1] += 1;
                    link["count"] = set[key1];
                    link["inverse"] = false;
                } else if (key2 in set) {
                    set[key2] += 1;
                    link["count"] = set[key2];
                    link["inverse"] = true;
                } else {
                    set[key1] = 1;
                    link["count"] = 1;
                    link["inverse"] = false;
                }
            }
        }
        return graphList;
    }

    data["graphs"] = calcLinkCount(data["graphs"]);
    data["coverage_max_precision"] = calcMatchingPatternsCount(data["coverage_max_precision"], data["graphs"]);
    return data
};


DataLoader.loadRunGen = function(r, g, startCB, resultCB, failCB, alwaysCB) {
    var fn;
    if (r < 0) {
        // a hack which will work for the next 984 years (2xxx)
        fn = 'results.json.gz';
    } else {
        r = r < 10 ? '0' + r : '' + r;
        if (g < 0 || g === '') {
            fn = 'results_run_' + r + '.json.gz';
        } else {
            g = g < 10 ? '0' + g : '' + g;
            fn = 'top_graph_patterns_run_' + r + '_gen_' + g + '.json.gz';

        }
    }
    return DataLoader.loadFile(fn, startCB, resultCB, failCB, alwaysCB);
};


if (!'SCRIPT_ROOT' in Window ||
    !'Zlib' in window) {
    console.error("Global var SCRIPT_ROOT must be set and module Zlib must be "+
        "loaded for module DataLoader to work.");
    delete DataLoader;
}
