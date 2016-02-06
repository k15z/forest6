QUnit.test("forest.predict", function( assert ) {
    var forest = new Forest([
        {
            index: 0,
            threshold: 2,
            left: {
                leaf: true,
                label: {a: 2, b: 1}
            },
            right: {
                index: 1,
                threshold: 1,
                left: {
                    leaf: true,
                    label: {a: 1}
                },
                right: {
                    leaf: true,
                    label: {b: 1}
                }
            }
        }
    ]);
    assert.equal(forest.predict([0,0])[0].label, "a");
    assert.equal(forest.predict([3,0])[0].label, "a");
    assert.equal(forest.predict([3,3])[0].label, "b");
    assert.notEqual(forest.predict([3,0])[0].label, "b");
    assert.notEqual(forest.predict([0,0])[0].label, "b");
    assert.notEqual(forest.predict([0,6])[1].label, "a");
});

QUnit.test("forest.train", function( assert ) {
    var forest = new Forest();
    forest.train([
        [0,0,1],
        [0,1,0],
        [0,1,1],
        [1,0,0],
        [1,0,1],
        [1,1,0]
    ], ["a", "b", "c","d","e","f"]);
    assert.equal(forest.predict([0,0,1])[0].label, "a");
    assert.equal(forest.predict([0,1,0])[0].label, "b");
    assert.equal(forest.predict([0,1,1])[0].label, "c");
    assert.equal(forest.predict([1,0,0])[0].label, "d");
    assert.equal(forest.predict([1,0,1])[0].label, "e");
    assert.equal(forest.predict([1,1,0])[0].label, "f");
});

QUnit.test("forest._util._randi", function( assert ) {
    var forest = new Forest();
    assert.ok(forest._util._randi(5) < 5);
    assert.ok(forest._util._randi(2, 5) >= 2 && forest._util._randi(2, 5) < 5);
    assert.ok(forest._util._randi(4, 5) >= 4 && forest._util._randi(4, 5) < 5);
});

QUnit.test("forest._util._range", function( assert ) {
    var forest = new Forest();
    assert.deepEqual(forest._util._range(5),  [0,1,2,3,4]);
    assert.deepEqual(forest._util._range(2, 6), [2,3,4,5]);
    assert.deepEqual(forest._util._range(-1, 2), [-1,0,1]);
});

QUnit.test("forest._util._count", function( assert ) {
    var forest = new Forest();
    assert.deepEqual(forest._util._count(["a", "a", "b"]), {a: 2, b: 1});
    assert.deepEqual(forest._util._count(["a", "a", "b"], [0, 1]), {a: 2});
    assert.deepEqual(forest._util._count(["a", "a", "b"], [1,2]), {a: 1, b: 1});
});

QUnit.test("forest._util._entropy", function( assert ) {
    var forest = new Forest();
    assert.ok(forest._util._entropy(["a", "a", "a"]) < forest._util._entropy(["a", "a", "b"]));
    assert.ok(forest._util._entropy(["a", "a", "b"]) == forest._util._entropy(["a", "b", "b"]));
    assert.ok(forest._util._entropy(["a", "a", "a"], [1,2]) < forest._util._entropy(["a", "a", "b"],[1,2]));
});
