//import React, { Component } from 'react';

class Table extends React.Component {

 constructor(props){
 super(props);
 this.getHeader = this.getHeader.bind(this);
 this.getRowsData = this.getRowsData.bind(this);
 this.getKeys = this.getKeys.bind(this);
 this.connectToWS = this.connectToWS.bind(this);
 this.sendMsg = this.sendMsg.bind(this);
 this.closeConn = this.closeConn.bind(this);
 this.myWebSocket;

 this.state = {
    data: this.props.data
 }
 this.connectToWS()

 }

 getKeys = function(){
  return Object.keys(this.state.data[0]);
  }

 getHeader = function(){
  var keys = this.getKeys();
  return keys.map((key, index)=>{
  return <th key={key}>{key.toUpperCase()}</th>
  })
  }

  getRowsData = function(){
      var items = this.state.data;
      var keys = this.getKeys();
      return items.map((row, index)=>{
        return <tr key={index}><RenderRow key={index} data={row} keys={keys}/></tr>
      })
  }


  connectToWS = function() {
      var endpoint = 'ws://localhost:8088/ws';
      if (this.myWebSocket !== undefined) {
          this.myWebSocket.close()
      }
      this.myWebSocket = new WebSocket(endpoint);
      this.myWebSocket.onmessage = (event) => {
          var leng;
          if (event.data.size === undefined) {
              leng = event.data.length
          } else {
              leng = event.data.size
          }
          console.log("onmessage. size: " + leng + ", content: " + event.data + ", time:" + (new Date()).getTime());
          //Updating table's state
          if(true) {
              this.setState((state, props) => {
                const newElement = {'Evolution': 'Apple', 'Generation': 100, 'Population': 'current', 'Stage': 'evaluation', 'Description': event.data};
                return {
                           data: state.data.concat(newElement)
                };
              });
          }
      };
      this.myWebSocket.onopen = function(evt) {
          console.log("onopen.");
      };
      this.myWebSocket.onclose = function(evt) {
          console.log("onclose.");
      };
      this.myWebSocket.onerror = function(evt) {
          console.log("Error!");
      };
  }

  sendMsg = function() {
      var message = document.getElementById("myMessage").value;
      this.myWebSocket.send(message);
      console.log("Time of sending message to server:" + (new Date()).getTime());
  }

  closeConn = function() {
      this.myWebSocket.close();
  }


 render() {
     return (
     <div>
     <table>
     <thead>
     <tr>{this.getHeader()}</tr>
     </thead>
     <tbody>
     {this.getRowsData()}
     </tbody>
     </table>
     </div>

     )
 }
}

const RenderRow = (props) =>{
 return props.keys.map((key, index)=>{
    var arr = [1,2,3,4,5]
    if(key == "Population") {
        //        var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
//        const detachedSVG = d3.create("svg");
//        detachedSVG.append('rect')
//          .attr('width', 40)
//          .attr('height', 40)
//          .attr('stroke', 'black')
//          .attr('fill', '#69a3b2');
        return <td key={props.data[key]}>
            <svg version="1.1" id="Capa_1" x="0px" y="0px" width="300px" height="50px">
                {
                arr.map((item, index)=>(
                <g>
                  <rect y="0" x={index*50} width="30" height="30"></rect>
                  <text y="0" x={index*50} font-family="Verdana" fontSize="35" fill="blue">{item}</text>
                </g>
                ))
                }
            </svg>
        </td>
    } else {
        return <td key={props.data[key]}>{props.data[key]}</td>
    }
 })
}

var dataObj = [ {'Evolution': 'Apple', 'Generation': 100, 'Population': 'current', 'Stage': 'evaluation', 'Description': 'Min/Max/Avg performance'} ]

const tableElement = <Table data={dataObj }/>;

const tableDOMContainer = document.getElementById('table_container');
ReactDOM.render(tableElement, tableDOMContainer);
