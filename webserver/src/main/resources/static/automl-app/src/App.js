import React, { Component } from 'react';

import RunDescription from "./js/components/RunDescription";
import Table from "./js/components/Table";
import TimeboxProgressBar from "./js/components/TimeboxProgressBar";
import Linechart from "./js/components/linechart/Linechart";


class App extends Component {

    constructor(props){
          super(props)
          this.wsHandler = this.wsHandler.bind(this)
          this.runDescriptionRef = React.createRef();
          this.appthis = this;
    }

    componentDidMount() {
          this.wsHandler()
    }
    componentDidUpdate() {
      this.wsHandler()
    }

    wsHandler() {
        var appthis = this
        var ag_global_vars = { }

        var connectToWS = function() {
          var endpoint = 'ws://localhost:8098/ws';
          if (ag_global_vars.myWebSocket !== undefined) {
              ag_global_vars.myWebSocket.close()
          }
          ag_global_vars.myWebSocket = new WebSocket(endpoint);
          ag_global_vars.myWebSocket.onmessage = (event) => {

              var serverDataJson = JSON.parse(event.data);

              var tableRelevantKeys = ["evolutionProgress", "population", "evaluatedTemplateData"]

              if(tableRelevantKeys.includes(serverDataJson.key)) {
                console.log("Forwarding data to tableRef")
                ag_global_vars.tableRef.update(serverDataJson)
              }
              else {
                console.log(JSON.stringify(serverDataJson))
                console.log("Update to other components")

              }

              if(serverDataJson.key == "evolutionProgress") {
                if(typeof serverDataJson.evolution !== 'undefined') {
                ag_global_vars.timeboxesRef.updateProgressBar(JSON.stringify(serverDataJson.evolution - 1))
                 console.log("Evolution:"+ JSON.stringify(serverDataJson.evolution))
                 console.log("Generation:"+ JSON.stringify(serverDataJson.generation))
                }
                console.log(JSON.stringify(serverDataJson))
              }
              if(serverDataJson.key == "timeboxesSetup") {
                console.log("Timeboxes:"+ JSON.stringify(serverDataJson.timeboxes))
                ag_global_vars.timeboxesRef.setupProgressBar(serverDataJson.timeboxes)
              }
              if(serverDataJson.key == "runDescription") {
                console.log("RunDescription:"+ JSON.stringify(serverDataJson.projectName))
                console.log("RunDescription:"+ JSON.stringify(serverDataJson.initialPopulationSize))
                ag_global_vars.runDescriptionRef.data = serverDataJson;
                //ag_global_vars.runDescriptionRef.setupComponent(serverDataJson)
              }


          };
          ag_global_vars.myWebSocket.onopen = function(evt) {
              console.log("onopen.");
          };
          ag_global_vars.myWebSocket.onclose = function(evt) {
              console.log("onclose.");
          };
          ag_global_vars.myWebSocket.onerror = function(evt) {
              console.log("Error!");
          };
        }

        var sendMsg = function() {
          var message = document.getElementById("myMessage").value;
          ag_global_vars.myWebSocket.send(message);
          console.log("Time of sending message to server:" + (new Date()).getTime());
        }

        var closeConn = function() {
          ag_global_vars.myWebSocket.close();
        }

        connectToWS()

        let counter = setInterval(timer , 1000)
        function timer() {
            appthis.runDescriptionRef.current.changeState();
            appthis.runDescriptionRef.current.setState(function(state, props) {
                                                                     return {
                                                                       counter: state.counter + 1
                                                                     };
                                                                   });


        }
    }

  render() {
        var dataObj = [ {'Evolution': 'Apple', 'Generation': 200, 'Population': 'current', 'Stage': 'evaluation', 'Description': 'Min/Max/Avg performance', 'evolutionProgress': {'evolution': "-1", 'generation': "-1"}} ]

        return (
        <div>
              <div className="jumbotron">
                  <div className="container">
                      <h1>AutoML Genetic</h1>
                  </div>
                  <div className="runDescription">
                  </div>
              </div>
              <div className="container-fluid">
                  <div className="row">
                      <div className="col-6 col-md-4" align="center">Fitness ( Phenotypic performance ) </div>
                      <div className="col-6 col-md-4">Complexity ( Avg. number of nodes per individual ) </div>
                      <div className="col-6 col-md-4">Gene frequencies</div>
                  </div>
                  <div className="row">
                      <div className="col-6 col-md-4">
                        <Linechart/>
                      </div>
                      <div className="col-6 col-md-4">Chart 2</div>
                      <div className="col-6 col-md-4">Chart 3</div>
                  </div>


                  <div className="row">
                      <div className="col-6 col-md-4">Repetitions of individuals during evolution</div>
                      <div className="col-6 col-md-4">Avg depth of individuals</div>
                      <div className="col-6 col-md-4">.col-6 .col-md-4</div>
                  </div>

                  <div className="row">
                      <div className="col-6">.col-6</div>
                      <div className="col-6">.col-6</div>
                  </div>
              </div>

              <TimeboxProgressBar />

              <Table data={dataObj}/>

              <RunDescription ref={this.runDescriptionRef}/>
            </div>
        );
  }
}
export default App;