const Konva = require('konva');
var request = require('xhr-request')

var stage = new Konva.Stage({
  container: 'container',
  width: 4000, // big enough to handle most images as there is little here on resizing and fitting
  height: 3000
});

window.scoreImage = function () {
  var imageUrl =  document.querySelector("#imageUrl").value;

  request('http://detectron2api.eastus.azurecontainer.io:5000/api/score-image', {
    method: 'POST',
    json: true,
    body: { imageUrl: imageUrl }
  }, function (err, data) {
    if (err)
    {
      alert("Failed to call web API!");
      console.error(error);
    }
    else
    {
      console.log(data);
      renderResult(imageUrl, data);
    }
  })

}

function renderResult(imageUrl, result) {
  stage.clear();
  var layer = new Konva.Layer();
  stage.add(layer);

  Konva.Image.fromURL(imageUrl, function (imageNode) {
    imageNode.setAttrs({
      x: 0,
      y: 0,
      scaleX: 1,
      scaleY: 1
    });
    layer.add(imageNode);
    result.pred_classes.forEach((item, index) => {
      var classLabel = result.classes[item];
      var boundingBox = result.pred_boxes[index];
      var score = (100 * result.scores[index]).toFixed();
      addObject(layer, classLabel, score, boundingBox[0], boundingBox[1], boundingBox[2], boundingBox[3]);
    });

    layer.batchDraw();

  });
}

function addObject(layer, label, score, x1, y1, x2, y2) {
  var boundingBox = new Konva.Rect({
    x: x1,
    y: y1,
    width: x2 - x1,
    height: y2 - y1,
    stroke: 'blue',
    strokeWidth: 3
  });

  var objectLabel = new Konva.Text({
    x: x1,
    y: y1,
    text: `${label} : ${score}%`,
    fontSize: 20,
    fontFamily: 'Calibri',
    fill: 'yellow'
  });

  layer.add(boundingBox);
  layer.add(objectLabel);
}




