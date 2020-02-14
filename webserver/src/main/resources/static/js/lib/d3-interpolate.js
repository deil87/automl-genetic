// https://d3js.org/d3-interpolate/ Version 1.1.6. Copyright 2017 Mike Bostock.
!function(t,n){"object"==typeof exports&&"undefined"!=typeof module?n(exports,require("d3-color")):"function"==typeof define&&define.amd?define(["exports","d3-color"],n):n(t.d3=t.d3||{},t.d3)}(this,function(t,n){"use strict";function r(t,n,r,e,o){var u=t*t,a=u*t;return((1-3*t+3*u-a)*n+(4-6*u+3*a)*r+(1+3*t+3*u-3*a)*e+a*o)/6}function e(t,n){return function(r){return t+r*n}}function o(t,n,r){return t=Math.pow(t,r),n=Math.pow(n,r)-t,r=1/r,function(e){return Math.pow(t+e*n,r)}}function u(t,n){var r=n-t;return r?e(t,r>180||r<-180?r-360*Math.round(r/360):r):Y(isNaN(t)?n:t)}function a(t){return 1==(t=+t)?i:function(n,r){return r-n?o(n,r,t):Y(isNaN(n)?r:n)}}function i(t,n){var r=n-t;return r?e(t,r):Y(isNaN(t)?n:t)}function l(t){return function(r){var e,o,u=r.length,a=new Array(u),i=new Array(u),l=new Array(u);for(e=0;e<u;++e)o=n.rgb(r[e]),a[e]=o.r||0,i[e]=o.g||0,l[e]=o.b||0;return a=t(a),i=t(i),l=t(l),o.opacity=1,function(t){return o.r=a(t),o.g=i(t),o.b=l(t),o+""}}}function c(t){return function(){return t}}function f(t){return function(n){return t(n)+""}}function s(t){return"none"===t?O:(M||(M=document.createElement("DIV"),w=document.documentElement,X=document.defaultView),M.style.transform=t,t=X.getComputedStyle(w.appendChild(M),null).getPropertyValue("transform"),w.removeChild(M),t=t.slice(7,-1).split(","),P(+t[0],+t[1],+t[2],+t[3],+t[4],+t[5]))}function p(t){return null==t?O:(A||(A=document.createElementNS("http://www.w3.org/2000/svg","g")),A.setAttribute("transform",t),(t=A.transform.baseVal.consolidate())?(t=t.matrix,P(t.a,t.b,t.c,t.d,t.e,t.f)):O)}function h(t,n,r,e){function o(t){return t.length?t.pop()+" ":""}function u(t,e,o,u,a,i){if(t!==o||e!==u){var l=a.push("translate(",null,n,null,r);i.push({i:l-4,x:E(t,o)},{i:l-2,x:E(e,u)})}else(o||u)&&a.push("translate("+o+n+u+r)}function a(t,n,r,u){t!==n?(t-n>180?n+=360:n-t>180&&(t+=360),u.push({i:r.push(o(r)+"rotate(",null,e)-2,x:E(t,n)})):n&&r.push(o(r)+"rotate("+n+e)}function i(t,n,r,u){t!==n?u.push({i:r.push(o(r)+"skewX(",null,e)-2,x:E(t,n)}):n&&r.push(o(r)+"skewX("+n+e)}function l(t,n,r,e,u,a){if(t!==r||n!==e){var i=u.push(o(u)+"scale(",null,",",null,")");a.push({i:i-4,x:E(t,r)},{i:i-2,x:E(n,e)})}else 1===r&&1===e||u.push(o(u)+"scale("+r+","+e+")")}return function(n,r){var e=[],o=[];return n=t(n),r=t(r),u(n.translateX,n.translateY,r.translateX,r.translateY,e,o),a(n.rotate,r.rotate,e,o),i(n.skewX,r.skewX,e,o),l(n.scaleX,n.scaleY,r.scaleX,r.scaleY,e,o),n=r=null,function(t){for(var n,r=-1,u=o.length;++r<u;)e[(n=o[r]).i]=n.x(t);return e.join("")}}}function g(t){return((t=Math.exp(t))+1/t)/2}function d(t){return((t=Math.exp(t))-1/t)/2}function y(t){return((t=Math.exp(2*t))-1)/(t+1)}function v(t){return function(r,e){var o=t((r=n.hsl(r)).h,(e=n.hsl(e)).h),u=i(r.s,e.s),a=i(r.l,e.l),l=i(r.opacity,e.opacity);return function(t){return r.h=o(t),r.s=u(t),r.l=a(t),r.opacity=l(t),r+""}}}function b(t,r){var e=i((t=n.lab(t)).l,(r=n.lab(r)).l),o=i(t.a,r.a),u=i(t.b,r.b),a=i(t.opacity,r.opacity);return function(n){return t.l=e(n),t.a=o(n),t.b=u(n),t.opacity=a(n),t+""}}function x(t){return function(r,e){var o=t((r=n.hcl(r)).h,(e=n.hcl(e)).h),u=i(r.c,e.c),a=i(r.l,e.l),l=i(r.opacity,e.opacity);return function(t){return r.h=o(t),r.c=u(t),r.l=a(t),r.opacity=l(t),r+""}}}function m(t){return function r(e){function o(r,o){var u=t((r=n.cubehelix(r)).h,(o=n.cubehelix(o)).h),a=i(r.s,o.s),l=i(r.l,o.l),c=i(r.opacity,o.opacity);return function(t){return r.h=u(t),r.s=a(t),r.l=l(Math.pow(t,e)),r.opacity=c(t),r+""}}return e=+e,o.gamma=r,o}(1)}var M,w,X,A,N=function(t){var n=t.length-1;return function(e){var o=e<=0?e=0:e>=1?(e=1,n-1):Math.floor(e*n),u=t[o],a=t[o+1],i=o>0?t[o-1]:2*u-a,l=o<n-1?t[o+2]:2*a-u;return r((e-o/n)*n,i,u,a,l)}},C=function(t){var n=t.length;return function(e){var o=Math.floor(((e%=1)<0?++e:e)*n),u=t[(o+n-1)%n],a=t[o%n],i=t[(o+1)%n],l=t[(o+2)%n];return r((e-o/n)*n,u,a,i,l)}},Y=function(t){return function(){return t}},j=function t(r){function e(t,r){var e=o((t=n.rgb(t)).r,(r=n.rgb(r)).r),u=o(t.g,r.g),a=o(t.b,r.b),l=i(t.opacity,r.opacity);return function(n){return t.r=e(n),t.g=u(n),t.b=a(n),t.opacity=l(n),t+""}}var o=a(r);return e.gamma=t,e}(1),q=l(N),k=l(C),R=function(t,n){var r,e=n?n.length:0,o=t?Math.min(e,t.length):0,u=new Array(o),a=new Array(e);for(r=0;r<o;++r)u[r]=L(t[r],n[r]);for(;r<e;++r)a[r]=n[r];return function(t){for(r=0;r<o;++r)a[r]=u[r](t);return a}},S=function(t,n){var r=new Date;return t=+t,n-=t,function(e){return r.setTime(t+n*e),r}},E=function(t,n){return t=+t,n-=t,function(r){return t+n*r}},I=function(t,n){var r,e={},o={};null!==t&&"object"==typeof t||(t={}),null!==n&&"object"==typeof n||(n={});for(r in n)r in t?e[r]=L(t[r],n[r]):o[r]=n[r];return function(t){for(r in e)o[r]=e[r](t);return o}},B=/[-+]?(?:\d+\.?\d*|\.?\d+)(?:[eE][-+]?\d+)?/g,D=new RegExp(B.source,"g"),H=function(t,n){var r,e,o,u=B.lastIndex=D.lastIndex=0,a=-1,i=[],l=[];for(t+="",n+="";(r=B.exec(t))&&(e=D.exec(n));)(o=e.index)>u&&(o=n.slice(u,o),i[a]?i[a]+=o:i[++a]=o),(r=r[0])===(e=e[0])?i[a]?i[a]+=e:i[++a]=e:(i[++a]=null,l.push({i:a,x:E(r,e)})),u=D.lastIndex;return u<n.length&&(o=n.slice(u),i[a]?i[a]+=o:i[++a]=o),i.length<2?l[0]?f(l[0].x):c(n):(n=l.length,function(t){for(var r,e=0;e<n;++e)i[(r=l[e]).i]=r.x(t);return i.join("")})},L=function(t,r){var e,o=typeof r;return null==r||"boolean"===o?Y(r):("number"===o?E:"string"===o?(e=n.color(r))?(r=e,j):H:r instanceof n.color?j:r instanceof Date?S:Array.isArray(r)?R:"function"!=typeof r.valueOf&&"function"!=typeof r.toString||isNaN(r)?I:E)(t,r)},T=function(t,n){return t=+t,n-=t,function(r){return Math.round(t+n*r)}},V=180/Math.PI,O={translateX:0,translateY:0,rotate:0,skewX:0,scaleX:1,scaleY:1},P=function(t,n,r,e,o,u){var a,i,l;return(a=Math.sqrt(t*t+n*n))&&(t/=a,n/=a),(l=t*r+n*e)&&(r-=t*l,e-=n*l),(i=Math.sqrt(r*r+e*e))&&(r/=i,e/=i,l/=i),t*e<n*r&&(t=-t,n=-n,l=-l,a=-a),{translateX:o,translateY:u,rotate:Math.atan2(n,t)*V,skewX:Math.atan(l)*V,scaleX:a,scaleY:i}},_=h(s,"px, ","px)","deg)"),z=h(p,", ",")",")"),Q=Math.SQRT2,Z=function(t,n){var r,e,o=t[0],u=t[1],a=t[2],i=n[0],l=n[1],c=n[2],f=i-o,s=l-u,p=f*f+s*s;if(p<1e-12)e=Math.log(c/a)/Q,r=function(t){return[o+t*f,u+t*s,a*Math.exp(Q*t*e)]};else{var h=Math.sqrt(p),v=(c*c-a*a+4*p)/(2*a*2*h),b=(c*c-a*a-4*p)/(2*c*2*h),x=Math.log(Math.sqrt(v*v+1)-v),m=Math.log(Math.sqrt(b*b+1)-b);e=(m-x)/Q,r=function(t){var n=t*e,r=g(x),i=a/(2*h)*(r*y(Q*n+x)-d(x));return[o+i*f,u+i*s,a*r/g(Q*n+x)]}}return r.duration=1e3*e,r},F=v(u),G=v(i),J=x(u),K=x(i),U=m(u),W=m(i),$=function(t,n){for(var r=new Array(n),e=0;e<n;++e)r[e]=t(e/(n-1));return r};t.interpolate=L,t.interpolateArray=R,t.interpolateBasis=N,t.interpolateBasisClosed=C,t.interpolateDate=S,t.interpolateNumber=E,t.interpolateObject=I,t.interpolateRound=T,t.interpolateString=H,t.interpolateTransformCss=_,t.interpolateTransformSvg=z,t.interpolateZoom=Z,t.interpolateRgb=j,t.interpolateRgbBasis=q,t.interpolateRgbBasisClosed=k,t.interpolateHsl=F,t.interpolateHslLong=G,t.interpolateLab=b,t.interpolateHcl=J,t.interpolateHclLong=K,t.interpolateCubehelix=U,t.interpolateCubehelixLong=W,t.quantize=$,Object.defineProperty(t,"__esModule",{value:!0})});