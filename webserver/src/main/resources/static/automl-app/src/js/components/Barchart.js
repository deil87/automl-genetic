import React, { Component } from 'react'
import ReactDOM from "react-dom"

import { scaleLinear } from 'd3-scale'
import { max } from 'd3-array'
import { select } from 'd3-selection'

class Barchart extends Component {
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
      const dataMax = max(this.props.data)
      const yScale = scaleLinear()
         .domain([0, dataMax])
         .range([0, this.props.size[1]])

      select(node)
         .selectAll('rect')
         .data(this.props.data)
         .enter()
         .append('rect')

      select(node)
         .selectAll('rect')
         .data(this.props.data)
         .exit()
         .remove()

      select(node)
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

export default Barchart;

const wrapper = document.getElementById("barchart_container");
wrapper ? ReactDOM.render(<Barchart data={[5,10,1,3, 4,15,9]} size={[500,500]}/>, wrapper) : false;