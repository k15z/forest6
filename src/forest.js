function Forest (model) {
    var model = model ? model : [];

    function predict (data) {
        var total = 0;
        var count = {};
        model.forEach(function(tree) {
            while (!tree.leaf)
                if (data[tree.index] < tree.threshold)
                    tree = tree.left;
                else
                    tree = tree.right;
            for (var key in tree.label)
                if (tree.label.hasOwnProperty(key)) {
                    if (!count[key])
                        count[key] = 0;
                    count[key] += tree.label[key];
                    total += tree.label[key];
                }
        });

        var probability = [];
        for (var key in count)
            if (count.hasOwnProperty(key))
                probability.push({"label": key, "probability": count[key]/total})
        probability.sort(function(a,b) {
            return b.probability - a.probability
        });
        return probability;
    }

    function train (data, label, options) {
        var options = {
            depth: ((options && options.depth) ? options.depth : 2),
            tries: ((options && options.tries) ? options.tries : 4),
            trees: ((options && options.trees) ? options.trees : 8)
        }

        model.length = 0;
        for (var m = 0; m < options.trees; m++) {
            var tree = {};
            tree.depth = options.depth;
            tree.all_ix = _range(data.length);

            var queue = [];
            queue.push(tree);
            while (queue.length > 0) {
                var node = queue.shift();
                if (node.depth == 0 || node.all_ix.length <= 1) {
                    node.leaf = true;
                    node.label = _count(label, node.all_ix);
                } else {
                    var bestGain = -1;
                    var entropy = _entropy(label, node.all_ix);
                    for (var t = 0; t < options.tries; t++) {
                        var index = _randi(data[0].length);
                        var threshold = (function() {
                            var k = Math.random();
                            var d1 = node.all_ix[_randi(node.all_ix.length)];
                            var d2 = node.all_ix[_randi(node.all_ix.length)];
                            if (d2 == d1) d2 = node.all_ix[_randi(node.all_ix.length)];
                            return data[d1][index]*k + data[d2][index]*(1-k);
                        })();

                        var left_ix = [];
                        var right_ix = [];
                        node.all_ix.forEach(function(i) {
                            if (data[i][index] < threshold)
                                left_ix.push(i);
                            else
                                right_ix.push(i);
                        })
                        for (var d = 0; d < data.length; d++)
                        var infoGain = entropy
                            - _entropy(label, left_ix)*(left_ix.length/node.all_ix.length)
                            - _entropy(label, right_ix)*(right_ix.length/node.all_ix.length);
                        if (infoGain > bestGain) {
                            node.index = index;
                            node.threshold = threshold;
                            node.left_ix = left_ix;
                            node.right_ix = right_ix;
                        }
                    }
                    node.left = {
                        depth: node.depth - 1,
                        all_ix: left_ix
                    }
                    node.right = {
                        depth: node.depth - 1,
                        all_ix: right_ix
                    }
                    queue.push(node.left);
                    queue.push(node.right);
                }
            }
            model.push(tree);
        }
    }

    function _randi (low, high) {
        if (high == undefined) {
            high = low;
            low = 0;
        }
        return Math.floor(Math.random() * (high - low) + low);
    }

    function _range (start, end) {
        if (end == undefined) {
            end = start;
            start = 0;
        }

        var arr = new Array(end - start);
        for (var i = start; i < end; i ++)
            arr[i-start] = i;
        return arr;
    }

    function _count (label, ix) {
        if (!ix)
            ix = _range(label.length)

        var count = {};
        ix.forEach(function(i) {
            if (!count[label[i]])
                count[label[i]] = 0;
            count[label[i]]++;
        });
        return count;
    }

    function _entropy (label, ix) {
        if (!ix)
            ix = _range(label.length)

        var total = 0;
        var count = _count(label, ix);
        for (var key in count)
            if (count.hasOwnProperty(key))
                total += count[key];

        var prob = 0.0;
        var entropy = 0.0;
        for (var key in count)
            if (count.hasOwnProperty(key)) {
                prob = count[key]/total;
                entropy += prob * Math.log2(prob);
            }
        return -entropy;
    }

    var exports = {_util: {}};
    exports.model = model;
    exports.train = train;
    exports.predict = predict;
    exports._util._randi = _randi;
    exports._util._range = _range;
    exports._util._count = _count;
    exports._util._entropy = _entropy;
    return exports;
}
