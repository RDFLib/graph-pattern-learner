/**
 * Created by rouven on 02.05.16.
 */

FingerprintView = "FingerprintView" in window ? FingerprintView : {};

FingerprintView.showGraphCB = null;
FingerprintView.selected = 0; // goes from 0 (accumulated) to n for n graph patterns

$(document).ready(function() {
    FingerprintView.CANVAS_CONTAINER = $('#fingerprints');
});

FingerprintView.reset = function() {
    FingerprintView.CANVAS_CONTAINER.html("");
};

FingerprintView.drawPrec = function(x, y, prec, height, canvas, drawWithImgData, oversample) {
    function getRGBFromPrec(prec, raw) {
        if (raw == null) raw = false;
        var rgb = Util.HSVtoRGB(206/180*3.1415, prec*.721, (1-prec)*.282+.718);
        return raw ? rgb : "rgb("+rgb.r + ", " + rgb.g + ", " + rgb.b + ")";
    }
    //console.info(x, y, prec, height);
    var ctx = canvas.getContext("2d");
    if (height == null) height = 1;
    if (drawWithImgData == null) drawWithImgData = false;
    if (oversample == null) oversample = 0;
    if (drawWithImgData) {
        var imgData = ctx.createImageData(1, height);
        var rgb = getRGBFromPrec(prec, true) ;
        for (var i = 0; i < imgData.data.length; i += 4) {
            imgData.data[i]   = rgb.r;
            imgData.data[i+1] = rgb.g;
            imgData.data[i+2] = rgb.b;
            imgData.data[i+3] = 255;
        }
        //console.info(rgb.r, rgb.g, rgb.b, 255);
        ctx.putImageData(imgData, x, y);
    } else {
        ctx.fillStyle = getRGBFromPrec(prec);
        ctx.fillRect(x - oversample/2, y, 1 + oversample, height);
    }
};

FingerprintView.drawGP = function(height, graphIndex, canvas, maxWidth, xOffset, oversample) {
    function nodesToId(nodes) {
        var res = [];
        for (var i=0; i < nodes.length; i++) {
            res.push(btoa(nodes[i]));
        }
        return res.join("#");
    }

    if (xOffset == null) xOffset = 0;

    var cmp = {};
    var cmp_ = FingerprintView.coverageMaxPrecision;
    var i;
    for (i=0; i < cmp_.length; i++) {
        // graphIndex === undefined means global accumulated precision
        var prec = graphIndex == null ? parseFloat(cmp_[i][1]) : 0;
        cmp[nodesToId(cmp_[i][0])] =  prec
    }
    if (graphIndex != null) {
        // showing specific graph pattern
        var pattern = FingerprintView.graphPatterns[graphIndex];
        var gtps = pattern["gtp_precisions"];
        for (i=0; i < gtps.length; i++) {
            cmp[nodesToId(gtps[i][0])] = gtps[i][1];
        }
    }

    if (maxWidth == null) maxWidth = cmp_.length;
    var trueWidth = $(canvas).width();

    // oversample makes pixels wider so we can still see them
    // essentially each line is meant to become at least 1 px, if element width
    // is wider than necessary oversample is meant to be 0.
    // if undesired, set oversample to 0 from externally...
    if (oversample == null) {
        oversample = Math.ceil(maxWidth / trueWidth) - 1;
    }

    i = 0;
    var key;
    for (key in cmp) {
        if (cmp[key] > 0) {
            FingerprintView.drawPrec(
                i % maxWidth + xOffset,
                Math.floor(i / maxWidth) * height,
                cmp[key],
                height,
                canvas,
                null,
                oversample
            );
        }
        i++;
    }
    // return the height used
    return (Math.floor((cmp_.length-1)/maxWidth)+1)*height
};

FingerprintView.selectGP = function(graphIndex) {
    FingerprintView.CANVAS_CONTAINER.find('canvas.selected').removeClass('selected');
    if (graphIndex != null) {
        FingerprintView.CANVAS_CONTAINER.find('canvas[data-idx="'+(graphIndex+1)+'"]').addClass('selected');
    } else {
        FingerprintView.CANVAS_CONTAINER.find('canvas[data-idx="0"]').addClass('selected');
    }
};

FingerprintView.update = function(coverageMaxPrecision, graphPatterns, graphIndex) {
    $(document).ready(function() {
        FingerprintView.reset();
        FingerprintView.coverageMaxPrecision = coverageMaxPrecision;
        FingerprintView.graphPatterns = graphPatterns;
        FingerprintView.selected = graphIndex == null ? 0 : graphIndex+1;
        var width = coverageMaxPrecision.length;
        for (var i = 0; i < graphPatterns.length + 1; i++) {
            var canvas = $('<canvas width="'+width+'" height="1" data-idx="'+i+'"></canvas>');
            FingerprintView.CANVAS_CONTAINER.append(canvas);
            if (i == 0) {
                FingerprintView.drawGP(1, null, canvas[0], null, null, 0);
            } else {
                FingerprintView.drawGP(1, i-1, canvas[0], null, null, 0);
            }
            canvas.click(function() {
                if (FingerprintView.showGraphCB != null) {
                    var idx = parseInt($(this).attr('data-idx'));
                    FingerprintView.showGraphCB(idx==0 ? null : idx-1);
                }
            });
        }
        FingerprintView.selectGP(graphIndex);
    });
};

FingerprintView.addFingerPrint = function(containerSelector, graphIndex) {
    if (FingerprintView.coverageMaxPrecision == null) {
        console.error('Call FingerprintView.update before FingerprintView.addFingerPrint.')
    }
    var width = FingerprintView.coverageMaxPrecision.length;
    var canvas = $('<canvas width="'+width+'" height="1"></canvas>');
    $(containerSelector).append(canvas);
    FingerprintView.drawGP(1, graphIndex, canvas[0]);
};

FingerprintView.init = function(coverageMaxPrecision, graphPatterns, showGraphCB) {
    FingerprintView.update(coverageMaxPrecision, graphPatterns);
    FingerprintView.showGraphCB = showGraphCB;
};

// FIXME: check for other modules.
