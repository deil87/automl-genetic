
class Timeboxes extends React.Component {

 constructor(props){
 super(props);
 this.getHeader = this.getHeader.bind(this);
 this.getRowsData = this.getRowsData.bind(this);
 this.getKeys = this.getKeys.bind(this);
 this.update = this.update.bind(this);

 this.state = {
    data: {
        population: this.props.data,
        evolutionProgress: {'evolution': "-1", 'generation': "-1"},
        currentPopulation: undefined
    }
 }

 }

 getKeys = function(){
  return Object.keys(this.state.data.population[0]).filter(function(key) {
                                                                       if (key === "evolutionProgress") {
                                                                         return false; // skip
                                                                       }
                                                                       return true;
                                                                     });
  }

 update = function(serverDataJson){
    console.log("Hello update!")
    //Updating table's state
    if(serverDataJson.key == "evolutionProgress") {
      console.log("Evolution progress: Evolution number:" + serverDataJson.evolution + ", Generation number:" + serverDataJson.generation);
      this.setState((state, props) => {
          const newProgress = serverDataJson;
          return {
                    data: {
                      population:  state.data.population,
                      evolutionProgress: newProgress,
                      currentPopulation:  state.data.currentPopulation
                    }
          };
      });
    } else if(serverDataJson.key == "population") {
//          console.log("Current population: " + JSON.stringify(serverDataJson));
      this.setState((state, props) => {
          return {
                data: {
                  currentPopulation:  serverDataJson,
                  population:  state.data.population,
                  evolutionProgress: state.data.evolutionProgress
                }
          };
      });
    } else {
        this.setState((state, props) => {
          const newElement = {'Evolution': 'Apple', 'Generation': 100, 'Population': 'current', 'Stage': 'evaluation',
           'Description': event.data,
           'evolutionProgress': state.data.evolutionProgress};
           console.log(" Updating population with newElement:" + newElement)
          return {
             data: {
                population:  state.data.population.concat(newElement),
                evolutionProgress:  state.data.evolutionProgress,
                currentPopulation:  state.data.currentPopulation
             }
          };
        });
    }
  }

  getHeader = function(){
  var keys = this.getKeys();
  return keys.map((key, index)=>{
  return <th key={key}>{key.toUpperCase()}</th>
  })
  }

  getRowsData = function(){
      var items = this.state.data.population;
//      var ep = this.state.data.evolutionProgress;
      var cp = this.state.data.currentPopulation;
      console.log("getRowsData: " + JSON.stringify(cp));
      var keys = this.getKeys();
      return items.map((row, index)=>{
        return <tr key={index}><RenderRow key={index} data={row} keys={keys} currentPopulation={cp}/></tr>
      })
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
    var evolutionProgress = props.data['evolutionProgress']
//    console.log("RenderRow: evolutionProgress:" + JSON.stringify(evolutionProgress))
    var evolutionNumber = evolutionProgress.evolution
    var generationNumber = evolutionProgress.generation
//    console.log("Props:" + JSON.stringify(props.data))
    console.log("currentPopulation:" + JSON.stringify(props.currentPopulation))
    var currentPopulation = typeof props.currentPopulation === 'undefined' ? [] : props.currentPopulation['individuals']
//    console.log("currentPopulation:" + JSON.stringify(currentPopulation))
//    console.log("EvolutionNumber:" + JSON.stringify(evolutionNumber))
    if(key == "Population") {
        return <td key={props.data[key]}>
            <svg version="1.1" id="Capa_1" x="0px" y="0px" width="500px" height="50px">
                {
                        currentPopulation.map((item, index)=>(
                        <g>
                          <rect y="10" x={index*80} width="30" height="30" fill="white" stroke="black" stroke-width="1" alt="item" onMouseOver={() => console.log('over:' + index) } onMouseOut={() => console.log('out' + index) }></rect>
                          <text y="10" x={index*80} fontSize="15" fill="blue">{item}</text>
                        </g>
                        ))
                }
            </svg>
        </td>
    } else if(key == "Evolution") {
        return <td key={props.data[key]}>{evolutionNumber}</td>
    } else if(key == "Generation") {
        return <td key={props.data[key]}>{generationNumber}</td>
    } else {
        return <td key={props.data[key]}>{props.data[key]}</td>
    }
 })
}

var dataObj = [ {'Evolution': 'Apple', 'Generation': 200, 'Population': 'current', 'Stage': 'evaluation', 'Description': 'Min/Max/Avg performance', 'evolutionProgress': {'evolution': "-1", 'generation': "-1"}} ]

const tableElement = <Table data={dataObj }/>;

const tableDOMContainer = document.getElementById('table_container');
ag_global_vars.tableRef = ReactDOM.render(tableElement, tableDOMContainer);
