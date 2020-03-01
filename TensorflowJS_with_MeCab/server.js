var express = require('express')
var app = express()
var path = require('path');
var bodyParser = require('body-parser');

require('@tensorflow/tfjs-node');
app.use(express.static(path.join(__dirname,'/public')));
app.use(bodyParser.urlencoded({extended:false}));

app.get('/', function(req, res){
    res.send('??')
})


app.listen(52273, function(){

    console.log("서버가 실행 되었습니다. \n http://127.0.0.1:52273");

});