const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const fs=require('node:fs');
const code=fs.readFileSync(require('node:path').join(__dirname,'../public/app.js'),'utf8');
function setup(getUserMedia){
 const elements=new Map(); const events={}; const workers=[]; let stops=0;
 const track={stop(){stops++},addEventListener(){}};
 const media={getTracks:()=>[track],getVideoTracks:()=>[track]};
 const el=id=>{if(!elements.has(id))elements.set(id,{disabled:id==='stop',textContent:'',dataset:{},style:{},hidden:false,addEventListener(name,fn){this[name]=fn},pause(){},play:async()=>{},readyState:2,currentTime:1});return elements.get(id)};
 class Worker{constructor(){workers.push(this)}postMessage(data){if(data.type==='init')queueMicrotask(()=>this.onmessage({data:{type:'ready'}}))}terminate(){this.terminated=true}}
 const context={document:{getElementById:el,body:{dataset:{}},addEventListener:(n,f)=>events[n]=f},window:{Worker,createImageBitmap(){},OffscreenCanvas(){},addEventListener:(n,f)=>events[n]=f},Worker,navigator:{mediaDevices:{getUserMedia:()=>getUserMedia?getUserMedia(media):Promise.resolve(media)}},isSecureContext:true,setTimeout,clearTimeout,performance,createImageBitmap:async()=>({close(){}})};
 vm.runInNewContext(code,context);
 return {el,events,workers,context,get stops(){return stops}};
}
test('repeat start/stop releases tracks and worker',async()=>{
 const s=setup();await s.el('start').click();assert.equal(s.el('camera-state').textContent,'Aktiv');s.el('stop').click();assert.equal(s.stops,1);assert.equal(s.workers[0].terminated,true);assert.equal(s.el('camera').srcObject,null);await s.el('start').click();s.context.document.hidden=true;s.events.visibilitychange();assert.equal(s.stops,2);assert.equal(s.el('start').disabled,false);
});
test('cancel pending permission stops late stream without starting worker',async()=>{
 let resolve;const s=setup(media=>new Promise(r=>resolve=()=>r(media)));const pending=s.el('start').click();s.el('stop').click();resolve();await pending;assert.equal(s.stops,1);assert.equal(s.workers.length,0);assert.equal(s.el('start').disabled,false);
});
test('permission denial is localized and retry remains available',async()=>{
 const s=setup(()=>Promise.reject({name:'NotAllowedError'}));await s.el('start').click();assert.equal(s.el('result').textContent,'Kameraya icazə verilməyib');assert.equal(s.el('start').disabled,false);assert.equal(s.el('stop').disabled,true);
});

test('uncertain words rejected, stable words accepted, stale word cleared',async()=>{
 const s=setup();await s.el('start').click();
 const send=(recognition,hands=1)=>s.workers[0].onmessage({data:{type:'result',hands,recognition}});
 send({label:'ANA',confidence:0.6,runnerUp:0.3});
 assert.equal(s.el('result').textContent,'Ehtimal edilən söz: ANA');
 const good={label:'ANA',confidence:0.96,runnerUp:0.02};
 send(good);send(good);assert.equal(s.el('result').textContent,'Ehtimal edilən söz: ANA');
 send(good);assert.equal(s.el('result').textContent,'ANA');
 send(good,0);assert.equal(s.el('result').textContent,'İşarəni tanımaq mümkün olmadı');
 send({...good,confidence:NaN});assert.equal(s.el('result').textContent,'İşarəni tanımaq mümkün olmadı');
 s.el('stop').click();
});
