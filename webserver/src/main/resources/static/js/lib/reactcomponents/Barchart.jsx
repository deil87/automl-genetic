define(function(require){

var d3Array = require('d3-array');
var d3Selection = require('d3-selection');
var d3Scale = require('d3-scale');

class Barchart extends React.Component {
   constructor(props){
      super(props)
      this.createTBProgressBar = this.createTBProgressBar.bind(this)
   }
   componentDidMount() {
      this.createTBProgressBar()
   }
   componentDidUpdate() {
      this.createTBProgressBar()
   }

   createTBProgressBar() {
      const node = this.node
      const dataMax = d3Array.max(this.props.data)
      const yScale = d3Scale.scaleLinear()
         .domain([0, dataMax])
         .range([0, this.props.size[1]])

      d3Selection.select(node)
         .selectAll('rect')
         .data(this.props.data)
         .enter()
         .append('rect')

      d3Selection.select(node)
         .selectAll('rect')
         .data(this.props.data)
         .exit()
         .remove()

      d3Selection.select(node)
         .selectAll('rect')
         .data(this.props.data)
         .style('fill', '#fe9922')
         .attr('x', (d,i) => i * 25)
         .attr('y', d => this.props.size[1] - yScale(d))
         .attr('height', d => yScale(d))
         .attr('width', 25)
   }

   render() {
      return <svg ref={node => this.node = node}
      width={500} height={500}>
      </svg>
   }
}

const timeboxesReactElement = <Barchart data={[5,10,1,3, 4,15,9]} size={[500,500]}/>;

const tableDOMContainer = document.getElementById('barchart_container');
ag_global_vars.barchartRef = ReactDOM.render(timeboxesReactElement, tableDOMContainer);

});