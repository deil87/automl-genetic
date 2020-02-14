define(function(require){

var d3Array = require('d3-array');
var d3Selection = require('d3-selection');
var d3Scale = require('d3-scale');

class Linechart extends React.Component {
   constructor(props){
      super(props)
      this.createLinechart = this.createLinechart.bind(this)
   }
   componentDidMount() {
      this.createLinechart()
   }
   componentDidUpdate() {
      this.createLinechart()
   }

   createLinechart() {
      const node = this.node



   }

   render() {
         const margin = { top: 30, right: 132, bottom: 30, left: 50 }
         const width = 960 - margin.left - margin.right
         const height = 500 - margin.top - margin.bottom

      return <svg ref={node => this.node = node}
      width={width} height={height}>
      <rect x="0" y="0" width={width} height={height}></rect>
      </svg>
   }
}

const tableElement = <Linechart/>;

const tableDOMContainer = document.getElementById('linechart_container');
ag_global_vars.linechartRef = ReactDOM.render(tableElement, tableDOMContainer);

});