/**
 * Created by rouven on 10.09.15.
 */

var Util = Util ? "Util" in window : {};

// copied from here:
// http://stackoverflow.com/questions/17242144/javascript-convert-hsb-hsv-color-to-rgb-accurately
Util.HSVtoRGB = function(h, s, v) {
    var r, g, b, i, f, p, q, t;
    if (arguments.length === 1) {
        s = h.s, v = h.v, h = h.h;
    }
    i = Math.floor(h * 6);
    f = h * 6 - i;
    p = v * (1 - s);
    q = v * (1 - f * s);
    t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0:
            r = v, g = t, b = p;
            break;
        case 1:
            r = q, g = v, b = p;
            break;
        case 2:
            r = p, g = v, b = t;
            break;
        case 3:
            r = p, g = q, b = v;
            break;
        case 4:
            r = t, g = p, b = v;
            break;
        case 5:
            r = v, g = p, b = q;
            break;
    }
    return {
        r: Math.round(r * 255),
        g: Math.round(g * 255),
        b: Math.round(b * 255)
    }
};


Util.getQueryParams = function(str) {
    return (str || document.location.search).replace(/(^\?)/,'').split("&").map(function(n){return n = n.split("="),this[n[0]] = n[1],this}.bind({}))[0];
};


Util.encodeHash = function (view, pattern) {
    view = view || 'graph';
    if (view.indexOf('_') >= 0) throw "Don't use '_' in view names";
    if (view.indexOf('#') == 0) view = view.slice(1);
    if (pattern != null) return "#" + view + "_" + (pattern-0+1);
    else return "#" + view
};
Util.decodeHash = function (hash) {
    hash = hash || window.location.hash;
    var re = new RegExp("#?([^_]+)_(.*)");
    var match = hash.match(re);
    if (match != null) return [match[1], match[2]-1];
    re = new RegExp("#?([^_]+)");
    match = hash.match(re);
    if (match != null) return match.slice(1, 2);
    return null
};

Util.replaceHash = function (hash, uri) {
    uri = uri || window.location.href;
    var idx = uri.lastIndexOf("#");
    if (idx < 0) {
        return uri + hash;
    } else {
        return uri.slice(0, idx) + hash;
    }
};

/* https://stackoverflow.com/questions/5999118/add-or-update-query-string-parameter */
Util.setQueryParams = function(key, value, uri) {
    uri = uri || document.location.href;
    var re = new RegExp("([?&])" + key + "=[^#&$]*?(&|$|#)", "i");
    var separator = uri.indexOf('?') !== -1 ? "&" : "?";
    if (uri.match(re)) {
        return uri.replace(re, '$1' + key + "=" + value + '$2');
    }
    else {
        var uriParts = uri.split('#');
        if (uriParts.length == 1) uriParts.push('');
        return uriParts[0] + separator + key + "=" + value + '#' + uriParts[1];
    }
};
