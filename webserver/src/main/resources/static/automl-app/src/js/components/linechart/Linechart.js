import React, { Component } from "react";
import ReactDOM from "react-dom";

import styles from './linechart.css';

class Linechart extends Component {
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

      return <svg className={styles.overlay} ref={node => this.node = node}
      width={width} height={height}>
      <rect x="0" y="0" width={width} height={height}></rect>
      </svg>
   }
}

const tableElement = <Linechart/>;

const tableDOMContainer = document.getElementById('linechart_container');
ag_global_vars.linechartRef = ReactDOM.render(tableElement, tableDOMContainer);
