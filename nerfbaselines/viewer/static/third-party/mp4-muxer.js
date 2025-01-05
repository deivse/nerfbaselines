/*
MIT License

Copyright (c) 2023 Vanilagy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Source: https://github.com/Vanilagy/mp4-muxer/blob/87c247d4791fa224970fc901952de48ec6badc6a/build/mp4-muxer.min.mjs
*/
var Ne=(t,e,r)=>{if(!e.has(t))throw TypeError("Cannot "+r)};var i=(t,e,r)=>(Ne(t,e,"read from private field"),r?r.call(t):e.get(t)),f=(t,e,r)=>{if(e.has(t))throw TypeError("Cannot add the same private member more than once");e instanceof WeakSet?e.add(t):e.set(t,r)},C=(t,e,r,s)=>(Ne(t,e,"write to private field"),s?s.call(t,r):e.set(t,r),r),Ye=(t,e,r,s)=>({set _(n){C(t,e,n,r)},get _(){return i(t,e,s)}}),p=(t,e,r)=>(Ne(t,e,"access private method"),r);var c=new Uint8Array(8),I=new DataView(c.buffer),w=t=>[(t%256+256)%256],b=t=>(I.setUint16(0,t,!1),[c[0],c[1]]),Je=t=>(I.setInt16(0,t,!1),[c[0],c[1]]),Pe=t=>(I.setUint32(0,t,!1),[c[1],c[2],c[3]]),u=t=>(I.setUint32(0,t,!1),[c[0],c[1],c[2],c[3]]),et=t=>(I.setInt32(0,t,!1),[c[0],c[1],c[2],c[3]]),R=t=>(I.setUint32(0,Math.floor(t/2**32),!1),I.setUint32(4,t,!1),[c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]]),ye=t=>(I.setInt16(0,2**8*t,!1),[c[0],c[1]]),z=t=>(I.setInt32(0,2**16*t,!1),[c[0],c[1],c[2],c[3]]),Fe=t=>(I.setInt32(0,2**30*t,!1),[c[0],c[1],c[2],c[3]]),E=(t,e=!1)=>{let r=Array(t.length).fill(null).map((s,n)=>t.charCodeAt(n));return e&&r.push(0),r},Q=t=>t&&t[t.length-1],Se=t=>{let e;for(let r of t)(!e||r.presentationTimestamp>e.presentationTimestamp)&&(e=r);return e},B=(t,e,r=!0)=>{let s=t*e;return r?Math.round(s):s},Le=t=>{let e=t*(Math.PI/180),r=Math.cos(e),s=Math.sin(e);return[r,s,0,-s,r,0,0,0,1]},je=Le(0),$e=t=>[z(t[0]),z(t[1]),Fe(t[2]),z(t[3]),z(t[4]),Fe(t[5]),z(t[6]),z(t[7]),Fe(t[8])],K=t=>!t||typeof t!="object"?t:Array.isArray(t)?t.map(K):Object.fromEntries(Object.entries(t).map(([e,r])=>[e,K(r)])),$=t=>t>=0&&t<2**32;var y=(t,e,r)=>({type:t,contents:e&&new Uint8Array(e.flat(10)),children:r}),g=(t,e,r,s,n)=>y(t,[w(e),Pe(r),s??[]],n),tt=t=>{let e=512;return t.fragmented?y("ftyp",[E("iso5"),u(e),E("iso5"),E("iso6"),E("mp41")]):y("ftyp",[E("isom"),u(e),E("isom"),t.holdsAvc?E("avc1"):[],E("mp41")])},ve=t=>({type:"mdat",largeSize:t}),rt=t=>({type:"free",size:t}),se=(t,e,r=!1)=>y("moov",null,[mt(e,t),...t.map(s=>pt(s,e)),r?Wt(t):null]),mt=(t,e)=>{let r=B(Math.max(0,...e.filter(a=>a.samples.length>0).map(a=>{let l=Se(a.samples);return l.presentationTimestamp+l.duration})),xe),s=Math.max(...e.map(a=>a.id))+1,n=!$(t)||!$(r),o=n?R:u;return g("mvhd",+n,0,[o(t),o(t),u(xe),o(r),z(1),ye(1),Array(10).fill(0),$e(je),Array(24).fill(0),u(s)])},pt=(t,e)=>y("trak",null,[ct(t,e),bt(t,e)]),ct=(t,e)=>{let r=Se(t.samples),s=B(r?r.presentationTimestamp+r.duration:0,xe),n=!$(e)||!$(s),o=n?R:u,a;return t.info.type==="video"?a=typeof t.info.rotation=="number"?Le(t.info.rotation):t.info.rotation:a=je,g("tkhd",+n,3,[o(e),o(e),u(t.id),u(0),o(s),Array(8).fill(0),b(0),b(0),ye(t.info.type==="audio"?1:0),b(0),$e(a),z(t.info.type==="video"?t.info.width:0),z(t.info.type==="video"?t.info.height:0)])},bt=(t,e)=>y("mdia",null,[Tt(t,e),gt(t.info.type==="video"?"vide":"soun"),Ct(t)]),Tt=(t,e)=>{let r=Se(t.samples),s=B(r?r.presentationTimestamp+r.duration:0,t.timescale),n=!$(e)||!$(s),o=n?R:u;return g("mdhd",+n,0,[o(e),o(e),u(t.timescale),o(s),b(21956),b(0)])},gt=t=>g("hdlr",0,0,[E("mhlr"),E(t),u(0),u(0),u(0),E("mp4-muxer-hdlr",!0)]),Ct=t=>y("minf",null,[t.info.type==="video"?wt():yt(),St(),kt(t)]),wt=()=>g("vmhd",0,1,[b(0),b(0),b(0),b(0)]),yt=()=>g("smhd",0,0,[b(0),b(0)]),St=()=>y("dinf",null,[xt()]),xt=()=>g("dref",0,0,[u(1)],[vt()]),vt=()=>g("url ",0,1),kt=t=>{let e=t.compositionTimeOffsetTable.length>1||t.compositionTimeOffsetTable.some(r=>r.sampleCompositionTimeOffset!==0);return y("stbl",null,[At(t),Ft(t),Pt(t),Lt(t),jt(t),$t(t),e?Ht(t):null])},At=t=>g("stsd",0,0,[u(1)],[t.info.type==="video"?Et(er[t.info.codec],t):Vt(rr[t.info.codec],t)]),Et=(t,e)=>y(t,[Array(6).fill(0),b(1),b(0),b(0),Array(12).fill(0),b(e.info.width),b(e.info.height),u(4718592),u(4718592),u(0),b(1),Array(32).fill(0),b(24),Je(65535)],[tr[e.info.codec](e),e.info.decoderConfig.colorSpace?Ut(e):null]),Bt={bt709:1,bt470bg:5,smpte170m:6},Ot={bt709:1,smpte170m:6,"iec61966-2-1":13},zt={rgb:0,bt709:1,bt470bg:5,smpte170m:6},Ut=t=>y("colr",[E("nclx"),b(Bt[t.info.decoderConfig.colorSpace.primaries]),b(Ot[t.info.decoderConfig.colorSpace.transfer]),b(zt[t.info.decoderConfig.colorSpace.matrix]),w((t.info.decoderConfig.colorSpace.fullRange?1:0)<<7)]),It=t=>t.info.decoderConfig&&y("avcC",[...new Uint8Array(t.info.decoderConfig.description)]),Dt=t=>t.info.decoderConfig&&y("hvcC",[...new Uint8Array(t.info.decoderConfig.description)]),Mt=t=>{if(!t.info.decoderConfig)return null;let e=t.info.decoderConfig;if(!e.colorSpace)throw new Error("'colorSpace' is required in the decoder config for VP9.");let r=e.codec.split("."),s=Number(r[1]),n=Number(r[2]),o=Number(r[3]),a=0,l=(o<<4)+(a<<1)+Number(e.colorSpace.fullRange),m=2,T=2,v=2;return g("vpcC",1,0,[w(s),w(n),w(l),w(m),w(T),w(v),b(0)])},Rt=()=>{let t=1,e=1,r=(t<<7)+e;return y("av1C",[r,0,0,0])},Vt=(t,e)=>y(t,[Array(6).fill(0),b(1),b(0),b(0),u(0),b(e.info.numberOfChannels),b(16),b(0),b(0),z(e.info.sampleRate)],[ir[e.info.codec](e)]),_t=t=>{let e=new Uint8Array(t.info.decoderConfig.description);return g("esds",0,0,[u(58753152),w(32+e.byteLength),b(1),w(0),u(75530368),w(18+e.byteLength),w(64),w(21),Pe(0),u(130071),u(130071),u(92307584),w(e.byteLength),...e,u(109084800),w(1),w(2)])},Nt=t=>{let e=3840,r=0,s=t.info.decoderConfig?.description;if(s){if(s.byteLength<18)throw new TypeError("Invalid decoder description provided for Opus; must be at least 18 bytes long.");let n=ArrayBuffer.isView(s)?new DataView(s.buffer,s.byteOffset,s.byteLength):new DataView(s);e=n.getUint16(10,!0),r=n.getInt16(14,!0)}return y("dOps",[w(0),w(t.info.numberOfChannels),b(e),u(t.info.sampleRate),ye(r),w(0)])},Ft=t=>g("stts",0,0,[u(t.timeToSampleTable.length),t.timeToSampleTable.map(e=>[u(e.sampleCount),u(e.sampleDelta)])]),Pt=t=>{if(t.samples.every(r=>r.type==="key"))return null;let e=[...t.samples.entries()].filter(([,r])=>r.type==="key");return g("stss",0,0,[u(e.length),e.map(([r])=>u(r+1))])},Lt=t=>g("stsc",0,0,[u(t.compactlyCodedChunkTable.length),t.compactlyCodedChunkTable.map(e=>[u(e.firstChunk),u(e.samplesPerChunk),u(1)])]),jt=t=>g("stsz",0,0,[u(0),u(t.samples.length),t.samples.map(e=>u(e.size))]),$t=t=>t.finalizedChunks.length>0&&Q(t.finalizedChunks).offset>=2**32?g("co64",0,0,[u(t.finalizedChunks.length),t.finalizedChunks.map(e=>R(e.offset))]):g("stco",0,0,[u(t.finalizedChunks.length),t.finalizedChunks.map(e=>u(e.offset))]),Ht=t=>g("ctts",0,0,[u(t.compositionTimeOffsetTable.length),t.compositionTimeOffsetTable.map(e=>[u(e.sampleCount),u(e.sampleCompositionTimeOffset)])]),Wt=t=>y("mvex",null,t.map(qt)),qt=t=>g("trex",0,0,[u(t.id),u(1),u(0),u(0),u(0)]),He=(t,e)=>y("moof",null,[Xt(t),...e.map(Gt)]),Xt=t=>g("mfhd",0,0,[u(t)]),it=t=>{let e=0,r=0,s=0,n=0,o=t.type==="delta";return r|=+o,o?e|=1:e|=2,e<<24|r<<16|s<<8|n},Gt=t=>y("traf",null,[Zt(t),Kt(t),Qt(t)]),Zt=t=>{let e=0;e|=8,e|=16,e|=32,e|=131072;let r=t.currentChunk.samples[1]??t.currentChunk.samples[0],s={duration:r.timescaleUnitsToNextSample,size:r.size,flags:it(r)};return g("tfhd",0,e,[u(t.id),u(s.duration),u(s.size),u(s.flags)])},Kt=t=>g("tfdt",1,0,[R(B(t.currentChunk.startTimestamp,t.timescale))]),Qt=t=>{let e=t.currentChunk.samples.map(D=>D.timescaleUnitsToNextSample),r=t.currentChunk.samples.map(D=>D.size),s=t.currentChunk.samples.map(it),n=t.currentChunk.samples.map(D=>B(D.presentationTimestamp-D.decodeTimestamp,t.timescale)),o=new Set(e),a=new Set(r),l=new Set(s),m=new Set(n),T=l.size===2&&s[0]!==s[1],v=o.size>1,L=a.size>1,ie=!T&&l.size>1,Qe=m.size>1||[...m].some(D=>D!==0),j=0;return j|=1,j|=4*+T,j|=256*+v,j|=512*+L,j|=1024*+ie,j|=2048*+Qe,g("trun",1,j,[u(t.currentChunk.samples.length),u(t.currentChunk.offset-t.currentChunk.moofOffset||0),T?u(s[0]):[],t.currentChunk.samples.map((D,we)=>[v?u(e[we]):[],L?u(r[we]):[],ie?u(s[we]):[],Qe?et(n[we]):[]])])},st=t=>y("mfra",null,[...t.map(Yt),Jt()]),Yt=(t,e)=>g("tfra",1,0,[u(t.id),u(63),u(t.finalizedChunks.length),t.finalizedChunks.map(s=>[R(B(s.startTimestamp,t.timescale)),R(s.moofOffset),u(e+1),u(1),u(1)])]),Jt=()=>g("mfro",0,0,[u(0)]),er={avc:"avc1",hevc:"hvc1",vp9:"vp09",av1:"av01"},tr={avc:It,hevc:Dt,vp9:Mt,av1:Rt},rr={aac:"mp4a",opus:"Opus"},ir={aac:_t,opus:Nt};var sr=Symbol("isTarget"),H=class{};sr;var ne=class extends H{constructor(){super(...arguments);this.buffer=null}},W=class extends H{constructor(r){super();this.options=r;if(typeof r!="object")throw new TypeError("StreamTarget requires an options object to be passed to its constructor.");if(r.onData){if(typeof r.onData!="function")throw new TypeError("options.onData, when provided, must be a function.");if(r.onData.length<2)throw new TypeError("options.onData, when provided, must be a function that takes in at least two arguments (data and position). Ignoring the position argument, which specifies the byte offset at which the data is to be written, can lead to broken outputs.")}if(r.chunked!==void 0&&typeof r.chunked!="boolean")throw new TypeError("options.chunked, when provided, must be a boolean.");if(r.chunkSize!==void 0&&(!Number.isInteger(r.chunkSize)||r.chunkSize<=0))throw new TypeError("options.chunkSize, when provided, must be a positive integer.")}},oe=class extends H{constructor(r,s){super();this.stream=r;this.options=s;if(!(r instanceof FileSystemWritableFileStream))throw new TypeError("FileSystemWritableFileStreamTarget requires a FileSystemWritableFileStream instance.");if(s!==void 0&&typeof s!="object")throw new TypeError("FileSystemWritableFileStreamTarget's options, when provided, must be an object.");if(s&&s.chunkSize!==void 0&&(!Number.isInteger(s.chunkSize)||s.chunkSize<=0))throw new TypeError("options.chunkSize, when provided, must be a positive integer")}};var V,q,ae=class{constructor(){this.pos=0;f(this,V,new Uint8Array(8));f(this,q,new DataView(i(this,V).buffer));this.offsets=new WeakMap}seek(e){this.pos=e}writeU32(e){i(this,q).setUint32(0,e,!1),this.write(i(this,V).subarray(0,4))}writeU64(e){i(this,q).setUint32(0,Math.floor(e/2**32),!1),i(this,q).setUint32(4,e,!1),this.write(i(this,V).subarray(0,8))}writeAscii(e){for(let r=0;r<e.length;r++)i(this,q).setUint8(r%8,e.charCodeAt(r)),r%8===7&&this.write(i(this,V));e.length%8!==0&&this.write(i(this,V).subarray(0,e.length%8))}writeBox(e){if(this.offsets.set(e,this.pos),e.contents&&!e.children)this.writeBoxHeader(e,e.size??e.contents.byteLength+8),this.write(e.contents);else{let r=this.pos;if(this.writeBoxHeader(e,0),e.contents&&this.write(e.contents),e.children)for(let o of e.children)o&&this.writeBox(o);let s=this.pos,n=e.size??s-r;this.seek(r),this.writeBoxHeader(e,n),this.seek(s)}}writeBoxHeader(e,r){this.writeU32(e.largeSize?1:r),this.writeAscii(e.type),e.largeSize&&this.writeU64(r)}measureBoxHeader(e){return 8+(e.largeSize?8:0)}patchBox(e){let r=this.pos;this.seek(this.offsets.get(e)),this.writeBox(e),this.seek(r)}measureBox(e){if(e.contents&&!e.children)return this.measureBoxHeader(e)+e.contents.byteLength;{let r=this.measureBoxHeader(e);if(e.contents&&(r+=e.contents.byteLength),e.children)for(let s of e.children)s&&(r+=this.measureBox(s));return r}}};V=new WeakMap,q=new WeakMap;var de,_,Y,J,he,We,Ae=class extends ae{constructor(r){super();f(this,he);f(this,de,void 0);f(this,_,new ArrayBuffer(2**16));f(this,Y,new Uint8Array(i(this,_)));f(this,J,0);C(this,de,r)}write(r){p(this,he,We).call(this,this.pos+r.byteLength),i(this,Y).set(r,this.pos),this.pos+=r.byteLength,C(this,J,Math.max(i(this,J),this.pos))}finalize(){p(this,he,We).call(this,this.pos),i(this,de).buffer=i(this,_).slice(0,Math.max(i(this,J),this.pos))}};de=new WeakMap,_=new WeakMap,Y=new WeakMap,J=new WeakMap,he=new WeakSet,We=function(r){let s=i(this,_).byteLength;for(;s<r;)s*=2;if(s===i(this,_).byteLength)return;let n=new ArrayBuffer(s),o=new Uint8Array(n);o.set(i(this,Y),0),C(this,_,n),C(this,Y,o)};var fe,N,ue=class extends ae{constructor(r){super();f(this,fe,void 0);f(this,N,[]);C(this,fe,r)}write(r){i(this,N).push({data:r.slice(),start:this.pos}),this.pos+=r.byteLength}flush(){if(i(this,N).length===0)return;let r=[],s=[...i(this,N)].sort((n,o)=>n.start-o.start);r.push({start:s[0].start,size:s[0].data.byteLength});for(let n=1;n<s.length;n++){let o=r[r.length-1],a=s[n];a.start<=o.start+o.size?o.size=Math.max(o.size,a.start+a.data.byteLength-o.start):r.push({start:a.start,size:a.data.byteLength})}for(let n of r){n.data=new Uint8Array(n.size);for(let o of i(this,N))n.start<=o.start&&o.start<n.start+n.size&&n.data.set(o.data,o.start-n.start);i(this,fe).options.onData?.(n.data,n.start)}i(this,N).length=0}finalize(){}};fe=new WeakMap,N=new WeakMap;var nr=2**24,or=2,me,O,k,pe,qe,Be,nt,Oe,ot,ee,ke,le=class extends ae{constructor(r){super();f(this,pe);f(this,Be);f(this,Oe);f(this,ee);f(this,me,void 0);f(this,O,void 0);f(this,k,[]);if(C(this,me,r),C(this,O,r.options?.chunkSize??nr),!Number.isInteger(i(this,O))||i(this,O)<2**10)throw new Error("Invalid StreamTarget options: chunkSize must be an integer not smaller than 1024.")}write(r){p(this,pe,qe).call(this,r,this.pos),p(this,ee,ke).call(this),this.pos+=r.byteLength}finalize(){p(this,ee,ke).call(this,!0)}};me=new WeakMap,O=new WeakMap,k=new WeakMap,pe=new WeakSet,qe=function(r,s){let n=i(this,k).findIndex(T=>T.start<=s&&s<T.start+i(this,O));n===-1&&(n=p(this,Oe,ot).call(this,s));let o=i(this,k)[n],a=s-o.start,l=r.subarray(0,Math.min(i(this,O)-a,r.byteLength));o.data.set(l,a);let m={start:a,end:a+l.byteLength};if(p(this,Be,nt).call(this,o,m),o.written[0].start===0&&o.written[0].end===i(this,O)&&(o.shouldFlush=!0),i(this,k).length>or){for(let T=0;T<i(this,k).length-1;T++)i(this,k)[T].shouldFlush=!0;p(this,ee,ke).call(this)}l.byteLength<r.byteLength&&p(this,pe,qe).call(this,r.subarray(l.byteLength),s+l.byteLength)},Be=new WeakSet,nt=function(r,s){let n=0,o=r.written.length-1,a=-1;for(;n<=o;){let l=Math.floor(n+(o-n+1)/2);r.written[l].start<=s.start?(n=l+1,a=l):o=l-1}for(r.written.splice(a+1,0,s),(a===-1||r.written[a].end<s.start)&&a++;a<r.written.length-1&&r.written[a].end>=r.written[a+1].start;)r.written[a].end=Math.max(r.written[a].end,r.written[a+1].end),r.written.splice(a+1,1)},Oe=new WeakSet,ot=function(r){let n={start:Math.floor(r/i(this,O))*i(this,O),data:new Uint8Array(i(this,O)),written:[],shouldFlush:!1};return i(this,k).push(n),i(this,k).sort((o,a)=>o.start-a.start),i(this,k).indexOf(n)},ee=new WeakSet,ke=function(r=!1){for(let s=0;s<i(this,k).length;s++){let n=i(this,k)[s];if(!(!n.shouldFlush&&!r)){for(let o of n.written)i(this,me).options.onData?.(n.data.subarray(o.start,o.end),n.start+o.start);i(this,k).splice(s--,1)}}};var Ee=class extends le{constructor(e){super(new W({onData:(r,s)=>e.stream.write({type:"write",data:r,position:s}),chunkSize:e.options?.chunkSize}))}};var xe=1e3,ar=["avc","hevc","vp9","av1"],ur=["aac","opus"],lr=2082844800,dr=["strict","offset","cross-track-offset"],d,h,be,A,x,S,X,G,Ue,F,P,te,Ie,at,De,ut,Me,lt,Re,dt,Ve,ht,Te,Ge,U,M,_e,ft,re,ze,ge,Ze,Z,ce,Ce,Ke,Xe=class{constructor(e){f(this,Ie);f(this,De);f(this,Me);f(this,Re);f(this,Ve);f(this,Te);f(this,U);f(this,_e);f(this,re);f(this,ge);f(this,Z);f(this,Ce);f(this,d,void 0);f(this,h,void 0);f(this,be,void 0);f(this,A,void 0);f(this,x,null);f(this,S,null);f(this,X,Math.floor(Date.now()/1e3)+lr);f(this,G,[]);f(this,Ue,1);f(this,F,[]);f(this,P,[]);f(this,te,!1);if(p(this,Ie,at).call(this,e),e.video=K(e.video),e.audio=K(e.audio),e.fastStart=K(e.fastStart),this.target=e.target,C(this,d,{firstTimestampBehavior:"strict",...e}),e.target instanceof ne)C(this,h,new Ae(e.target));else if(e.target instanceof W)C(this,h,e.target.options?.chunked?new le(e.target):new ue(e.target));else if(e.target instanceof oe)C(this,h,new Ee(e.target));else throw new Error(`Invalid target: ${e.target}`);p(this,Re,dt).call(this),p(this,De,ut).call(this)}addVideoChunk(e,r,s,n){if(!(e instanceof EncodedVideoChunk))throw new TypeError("addVideoChunk's first argument (sample) must be of type EncodedVideoChunk.");if(r&&typeof r!="object")throw new TypeError("addVideoChunk's second argument (meta), when provided, must be an object.");if(s!==void 0&&(!Number.isFinite(s)||s<0))throw new TypeError("addVideoChunk's third argument (timestamp), when provided, must be a non-negative real number.");if(n!==void 0&&!Number.isFinite(n))throw new TypeError("addVideoChunk's fourth argument (compositionTimeOffset), when provided, must be a real number.");let o=new Uint8Array(e.byteLength);e.copyTo(o),this.addVideoChunkRaw(o,e.type,s??e.timestamp,e.duration,r,n)}addVideoChunkRaw(e,r,s,n,o,a){if(!(e instanceof Uint8Array))throw new TypeError("addVideoChunkRaw's first argument (data) must be an instance of Uint8Array.");if(r!=="key"&&r!=="delta")throw new TypeError("addVideoChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(s)||s<0)throw new TypeError("addVideoChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(n)||n<0)throw new TypeError("addVideoChunkRaw's fourth argument (duration) must be a non-negative real number.");if(o&&typeof o!="object")throw new TypeError("addVideoChunkRaw's fifth argument (meta), when provided, must be an object.");if(a!==void 0&&!Number.isFinite(a))throw new TypeError("addVideoChunkRaw's sixth argument (compositionTimeOffset), when provided, must be a real number.");if(p(this,Ce,Ke).call(this),!i(this,d).video)throw new Error("No video track declared.");if(typeof i(this,d).fastStart=="object"&&i(this,x).samples.length===i(this,d).fastStart.expectedVideoChunks)throw new Error(`Cannot add more video chunks than specified in 'fastStart' (${i(this,d).fastStart.expectedVideoChunks}).`);let l=p(this,Te,Ge).call(this,i(this,x),e,r,s,n,o,a);if(i(this,d).fastStart==="fragmented"&&i(this,S)){for(;i(this,P).length>0&&i(this,P)[0].decodeTimestamp<=l.decodeTimestamp;){let m=i(this,P).shift();p(this,U,M).call(this,i(this,S),m)}l.decodeTimestamp<=i(this,S).lastDecodeTimestamp?p(this,U,M).call(this,i(this,x),l):i(this,F).push(l)}else p(this,U,M).call(this,i(this,x),l)}addAudioChunk(e,r,s){if(!(e instanceof EncodedAudioChunk))throw new TypeError("addAudioChunk's first argument (sample) must be of type EncodedAudioChunk.");if(r&&typeof r!="object")throw new TypeError("addAudioChunk's second argument (meta), when provided, must be an object.");if(s!==void 0&&(!Number.isFinite(s)||s<0))throw new TypeError("addAudioChunk's third argument (timestamp), when provided, must be a non-negative real number.");let n=new Uint8Array(e.byteLength);e.copyTo(n),this.addAudioChunkRaw(n,e.type,s??e.timestamp,e.duration,r)}addAudioChunkRaw(e,r,s,n,o){if(!(e instanceof Uint8Array))throw new TypeError("addAudioChunkRaw's first argument (data) must be an instance of Uint8Array.");if(r!=="key"&&r!=="delta")throw new TypeError("addAudioChunkRaw's second argument (type) must be either 'key' or 'delta'.");if(!Number.isFinite(s)||s<0)throw new TypeError("addAudioChunkRaw's third argument (timestamp) must be a non-negative real number.");if(!Number.isFinite(n)||n<0)throw new TypeError("addAudioChunkRaw's fourth argument (duration) must be a non-negative real number.");if(o&&typeof o!="object")throw new TypeError("addAudioChunkRaw's fifth argument (meta), when provided, must be an object.");if(p(this,Ce,Ke).call(this),!i(this,d).audio)throw new Error("No audio track declared.");if(typeof i(this,d).fastStart=="object"&&i(this,S).samples.length===i(this,d).fastStart.expectedAudioChunks)throw new Error(`Cannot add more audio chunks than specified in 'fastStart' (${i(this,d).fastStart.expectedAudioChunks}).`);let a=p(this,Te,Ge).call(this,i(this,S),e,r,s,n,o);if(i(this,d).fastStart==="fragmented"&&i(this,x)){for(;i(this,F).length>0&&i(this,F)[0].decodeTimestamp<=a.decodeTimestamp;){let l=i(this,F).shift();p(this,U,M).call(this,i(this,x),l)}a.decodeTimestamp<=i(this,x).lastDecodeTimestamp?p(this,U,M).call(this,i(this,S),a):i(this,P).push(a)}else p(this,U,M).call(this,i(this,S),a)}finalize(){if(i(this,te))throw new Error("Cannot finalize a muxer more than once.");if(i(this,d).fastStart==="fragmented"){for(let r of i(this,F))p(this,U,M).call(this,i(this,x),r);for(let r of i(this,P))p(this,U,M).call(this,i(this,S),r);p(this,ge,Ze).call(this,!1)}else i(this,x)&&p(this,re,ze).call(this,i(this,x)),i(this,S)&&p(this,re,ze).call(this,i(this,S));let e=[i(this,x),i(this,S)].filter(Boolean);if(i(this,d).fastStart==="in-memory"){let r;for(let n=0;n<2;n++){let o=se(e,i(this,X)),a=i(this,h).measureBox(o);r=i(this,h).measureBox(i(this,A));let l=i(this,h).pos+a+r;for(let m of i(this,G)){m.offset=l;for(let{data:T}of m.samples)l+=T.byteLength,r+=T.byteLength}if(l<2**32)break;r>=2**32&&(i(this,A).largeSize=!0)}let s=se(e,i(this,X));i(this,h).writeBox(s),i(this,A).size=r,i(this,h).writeBox(i(this,A));for(let n of i(this,G))for(let o of n.samples)i(this,h).write(o.data),o.data=null}else if(i(this,d).fastStart==="fragmented"){let r=i(this,h).pos,s=st(e);i(this,h).writeBox(s);let n=i(this,h).pos-r;i(this,h).seek(i(this,h).pos-4),i(this,h).writeU32(n)}else{let r=i(this,h).offsets.get(i(this,A)),s=i(this,h).pos-r;i(this,A).size=s,i(this,A).largeSize=s>=2**32,i(this,h).patchBox(i(this,A));let n=se(e,i(this,X));if(typeof i(this,d).fastStart=="object"){i(this,h).seek(i(this,be)),i(this,h).writeBox(n);let o=r-i(this,h).pos;i(this,h).writeBox(rt(o))}else i(this,h).writeBox(n)}p(this,Z,ce).call(this),i(this,h).finalize(),C(this,te,!0)}};d=new WeakMap,h=new WeakMap,be=new WeakMap,A=new WeakMap,x=new WeakMap,S=new WeakMap,X=new WeakMap,G=new WeakMap,Ue=new WeakMap,F=new WeakMap,P=new WeakMap,te=new WeakMap,Ie=new WeakSet,at=function(e){if(typeof e!="object")throw new TypeError("The muxer requires an options object to be passed to its constructor.");if(!(e.target instanceof H))throw new TypeError("The target must be provided and an instance of Target.");if(e.video){if(!ar.includes(e.video.codec))throw new TypeError(`Unsupported video codec: ${e.video.codec}`);if(!Number.isInteger(e.video.width)||e.video.width<=0)throw new TypeError(`Invalid video width: ${e.video.width}. Must be a positive integer.`);if(!Number.isInteger(e.video.height)||e.video.height<=0)throw new TypeError(`Invalid video height: ${e.video.height}. Must be a positive integer.`);let r=e.video.rotation;if(typeof r=="number"&&![0,90,180,270].includes(r))throw new TypeError(`Invalid video rotation: ${r}. Has to be 0, 90, 180 or 270.`);if(Array.isArray(r)&&(r.length!==9||r.some(s=>typeof s!="number")))throw new TypeError(`Invalid video transformation matrix: ${r.join()}`);if(e.video.frameRate!==void 0&&(!Number.isInteger(e.video.frameRate)||e.video.frameRate<=0))throw new TypeError(`Invalid video frame rate: ${e.video.frameRate}. Must be a positive integer.`)}if(e.audio){if(!ur.includes(e.audio.codec))throw new TypeError(`Unsupported audio codec: ${e.audio.codec}`);if(!Number.isInteger(e.audio.numberOfChannels)||e.audio.numberOfChannels<=0)throw new TypeError(`Invalid number of audio channels: ${e.audio.numberOfChannels}. Must be a positive integer.`);if(!Number.isInteger(e.audio.sampleRate)||e.audio.sampleRate<=0)throw new TypeError(`Invalid audio sample rate: ${e.audio.sampleRate}. Must be a positive integer.`)}if(e.firstTimestampBehavior&&!dr.includes(e.firstTimestampBehavior))throw new TypeError(`Invalid first timestamp behavior: ${e.firstTimestampBehavior}`);if(typeof e.fastStart=="object"){if(e.video){if(e.fastStart.expectedVideoChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedVideoChunks'.");if(!Number.isInteger(e.fastStart.expectedVideoChunks)||e.fastStart.expectedVideoChunks<0)throw new TypeError("'expectedVideoChunks' must be a non-negative integer.")}if(e.audio){if(e.fastStart.expectedAudioChunks===void 0)throw new TypeError("'fastStart' is an object but is missing property 'expectedAudioChunks'.");if(!Number.isInteger(e.fastStart.expectedAudioChunks)||e.fastStart.expectedAudioChunks<0)throw new TypeError("'expectedAudioChunks' must be a non-negative integer.")}}else if(![!1,"in-memory","fragmented"].includes(e.fastStart))throw new TypeError("'fastStart' option must be false, 'in-memory', 'fragmented' or an object.")},De=new WeakSet,ut=function(){if(i(this,h).writeBox(tt({holdsAvc:i(this,d).video?.codec==="avc",fragmented:i(this,d).fastStart==="fragmented"})),C(this,be,i(this,h).pos),i(this,d).fastStart==="in-memory")C(this,A,ve(!1));else if(i(this,d).fastStart!=="fragmented"){if(typeof i(this,d).fastStart=="object"){let e=p(this,Me,lt).call(this);i(this,h).seek(i(this,h).pos+e)}C(this,A,ve(!0)),i(this,h).writeBox(i(this,A))}p(this,Z,ce).call(this)},Me=new WeakSet,lt=function(){if(typeof i(this,d).fastStart!="object")return;let e=0,r=[i(this,d).fastStart.expectedVideoChunks,i(this,d).fastStart.expectedAudioChunks];for(let s of r)s&&(e+=(4+4)*Math.ceil(2/3*s),e+=4*s,e+=(4+4+4)*Math.ceil(2/3*s),e+=4*s,e+=8*s);return e+=4096,e},Re=new WeakSet,dt=function(){if(i(this,d).video&&C(this,x,{id:1,info:{type:"video",codec:i(this,d).video.codec,width:i(this,d).video.width,height:i(this,d).video.height,rotation:i(this,d).video.rotation??0,decoderConfig:null},timescale:i(this,d).video.frameRate??57600,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,d).audio&&(C(this,S,{id:i(this,d).video?2:1,info:{type:"audio",codec:i(this,d).audio.codec,numberOfChannels:i(this,d).audio.numberOfChannels,sampleRate:i(this,d).audio.sampleRate,decoderConfig:null},timescale:i(this,d).audio.sampleRate,samples:[],finalizedChunks:[],currentChunk:null,firstDecodeTimestamp:void 0,lastDecodeTimestamp:-1,timeToSampleTable:[],compositionTimeOffsetTable:[],lastTimescaleUnits:null,lastSample:null,compactlyCodedChunkTable:[]}),i(this,d).audio.codec==="aac")){let e=p(this,Ve,ht).call(this,2,i(this,d).audio.sampleRate,i(this,d).audio.numberOfChannels);i(this,S).info.decoderConfig={codec:i(this,d).audio.codec,description:e,numberOfChannels:i(this,d).audio.numberOfChannels,sampleRate:i(this,d).audio.sampleRate}}},Ve=new WeakSet,ht=function(e,r,s){let o=[96e3,88200,64e3,48e3,44100,32e3,24e3,22050,16e3,12e3,11025,8e3,7350].indexOf(r),a=s,l="";l+=e.toString(2).padStart(5,"0"),l+=o.toString(2).padStart(4,"0"),o===15&&(l+=r.toString(2).padStart(24,"0")),l+=a.toString(2).padStart(4,"0");let m=Math.ceil(l.length/8)*8;l=l.padEnd(m,"0");let T=new Uint8Array(l.length/8);for(let v=0;v<l.length;v+=8)T[v/8]=parseInt(l.slice(v,v+8),2);return T},Te=new WeakSet,Ge=function(e,r,s,n,o,a,l){let m=n/1e6,T=(n-(l??0))/1e6,v=o/1e6,L=p(this,_e,ft).call(this,m,T,e);return m=L.presentationTimestamp,T=L.decodeTimestamp,a?.decoderConfig&&(e.info.decoderConfig===null?e.info.decoderConfig=a.decoderConfig:Object.assign(e.info.decoderConfig,a.decoderConfig)),{presentationTimestamp:m,decodeTimestamp:T,duration:v,data:r,size:r.byteLength,type:s,timescaleUnitsToNextSample:B(v,e.timescale)}},U=new WeakSet,M=function(e,r){i(this,d).fastStart!=="fragmented"&&e.samples.push(r);let s=B(r.presentationTimestamp-r.decodeTimestamp,e.timescale);if(e.lastTimescaleUnits!==null){let o=B(r.decodeTimestamp,e.timescale,!1),a=Math.round(o-e.lastTimescaleUnits);if(e.lastTimescaleUnits+=a,e.lastSample.timescaleUnitsToNextSample=a,i(this,d).fastStart!=="fragmented"){let l=Q(e.timeToSampleTable);l.sampleCount===1?(l.sampleDelta=a,l.sampleCount++):l.sampleDelta===a?l.sampleCount++:(l.sampleCount--,e.timeToSampleTable.push({sampleCount:2,sampleDelta:a}));let m=Q(e.compositionTimeOffsetTable);m.sampleCompositionTimeOffset===s?m.sampleCount++:e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:s})}}else e.lastTimescaleUnits=0,i(this,d).fastStart!=="fragmented"&&(e.timeToSampleTable.push({sampleCount:1,sampleDelta:B(r.duration,e.timescale)}),e.compositionTimeOffsetTable.push({sampleCount:1,sampleCompositionTimeOffset:s}));e.lastSample=r;let n=!1;if(!e.currentChunk)n=!0;else{let o=r.presentationTimestamp-e.currentChunk.startTimestamp;if(i(this,d).fastStart==="fragmented"){let a=i(this,x)??i(this,S);e===a&&r.type==="key"&&o>=1&&(n=!0,p(this,ge,Ze).call(this))}else n=o>=.5}n&&(e.currentChunk&&p(this,re,ze).call(this,e),e.currentChunk={startTimestamp:r.presentationTimestamp,samples:[]}),e.currentChunk.samples.push(r)},_e=new WeakSet,ft=function(e,r,s){let n=i(this,d).firstTimestampBehavior==="strict",o=s.lastDecodeTimestamp===-1;if(n&&o&&r!==0)throw new Error(`The first chunk for your media track must have a timestamp of 0 (received DTS=${r}).Non-zero first timestamps are often caused by directly piping frames or audio data from a MediaStreamTrack into the encoder. Their timestamps are typically relative to the age of thedocument, which is probably what you want.

If you want to offset all timestamps of a track such that the first one is zero, set firstTimestampBehavior: 'offset' in the options.
`);if(i(this,d).firstTimestampBehavior==="offset"||i(this,d).firstTimestampBehavior==="cross-track-offset"){s.firstDecodeTimestamp===void 0&&(s.firstDecodeTimestamp=r);let l;i(this,d).firstTimestampBehavior==="offset"?l=s.firstDecodeTimestamp:l=Math.min(i(this,x)?.firstDecodeTimestamp??1/0,i(this,S)?.firstDecodeTimestamp??1/0),r-=l,e-=l}if(r<s.lastDecodeTimestamp)throw new Error(`Timestamps must be monotonically increasing (DTS went from ${s.lastDecodeTimestamp*1e6} to ${r*1e6}).`);return s.lastDecodeTimestamp=r,{presentationTimestamp:e,decodeTimestamp:r}},re=new WeakSet,ze=function(e){if(i(this,d).fastStart==="fragmented")throw new Error("Can't finalize individual chunks if 'fastStart' is set to 'fragmented'.");if(e.currentChunk){if(e.finalizedChunks.push(e.currentChunk),i(this,G).push(e.currentChunk),(e.compactlyCodedChunkTable.length===0||Q(e.compactlyCodedChunkTable).samplesPerChunk!==e.currentChunk.samples.length)&&e.compactlyCodedChunkTable.push({firstChunk:e.finalizedChunks.length,samplesPerChunk:e.currentChunk.samples.length}),i(this,d).fastStart==="in-memory"){e.currentChunk.offset=0;return}e.currentChunk.offset=i(this,h).pos;for(let r of e.currentChunk.samples)i(this,h).write(r.data),r.data=null;p(this,Z,ce).call(this)}},ge=new WeakSet,Ze=function(e=!0){if(i(this,d).fastStart!=="fragmented")throw new Error("Can't finalize a fragment unless 'fastStart' is set to 'fragmented'.");let r=[i(this,x),i(this,S)].filter(m=>m&&m.currentChunk);if(r.length===0)return;let s=Ye(this,Ue)._++;if(s===1){let m=se(r,i(this,X),!0);i(this,h).writeBox(m)}let n=i(this,h).pos,o=He(s,r);i(this,h).writeBox(o);{let m=ve(!1),T=0;for(let L of r)for(let ie of L.currentChunk.samples)T+=ie.size;let v=i(this,h).measureBox(m)+T;v>=2**32&&(m.largeSize=!0,v=i(this,h).measureBox(m)+T),m.size=v,i(this,h).writeBox(m)}for(let m of r){m.currentChunk.offset=i(this,h).pos,m.currentChunk.moofOffset=n;for(let T of m.currentChunk.samples)i(this,h).write(T.data),T.data=null}let a=i(this,h).pos;i(this,h).seek(i(this,h).offsets.get(o));let l=He(s,r);i(this,h).writeBox(l),i(this,h).seek(a);for(let m of r)m.finalizedChunks.push(m.currentChunk),i(this,G).push(m.currentChunk),m.currentChunk=null;e&&p(this,Z,ce).call(this)},Z=new WeakSet,ce=function(){i(this,h)instanceof ue&&i(this,h).flush()},Ce=new WeakSet,Ke=function(){if(i(this,te))throw new Error("Cannot add new video or audio chunks after the file has been finalized.")};export{ne as ArrayBufferTarget,oe as FileSystemWritableFileStreamTarget,Xe as Muxer,W as StreamTarget};
