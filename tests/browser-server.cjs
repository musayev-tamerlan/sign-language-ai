const express = require('express');
const app = express();
app.get('/smoke',(_req,res)=>res.sendFile(__dirname+'/smoke.html'));
app.get('/smoke.js',(_req,res)=>res.sendFile(__dirname+'/smoke.js'));
app.use(require('../server'));
app.listen(3101,'127.0.0.1',()=>console.log('Test server on 3101'));
