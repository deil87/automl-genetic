import React, { Component } from 'react'
import ReactDOM from "react-dom"

import { select } from 'd3-selection'
import $ from "jquery"
import * as d3 from "d3"
//import ErrorBoundary from "./ErrorBoundary";

class TimeboxProgressBar extends Component {
   constructor(props){
      super(props)
      this.createTBProgressBar = this.createTBProgressBar.bind(this);
      this.setupProgressBar = this.setupProgressBar.bind(this);
      this.updateProgressBar = this.updateProgressBar.bind(this);

      this.colors = { green: '#4DC87F', lightGreen: '#D9F0E3' }

      this.setupHappened = false
   }

   componentDidMount() {
      this.createTBProgressBar()
   }
   componentDidUpdate() {
      this.createTBProgressBar()
   }

   setupProgressBar(steps) {
     this.steps = steps.map(function(x,i){ return i.toString()})
     console.log(this.steps)
     this.width = 1300
     this.height = 40
     this.offset = 48
     this.stepWidth = (this.width - this.offset * 2) / (this.steps.length - 1)
     this.setupHappened = true
     this.createTBProgressBar()
   }

   createTBProgressBar() {
   if(this.setupHappened) {
      const node = this.node

      var width = this.width
      var height = this.height
      var offset = this.offset
      var stepWidth = this.stepWidth

      var radius = 10

      const self = this

      width += offset * 2;
//      height += offset * 2;
      var dimensions = '' + 0 + ' ' + 0 + ' ' + width + ' ' + height;

      var svg = d3.select("#timeboxes_container")
              .append("svg")
              .attr('id', 'scene', true)
              .attr('preserveAspectRatio', 'xMinYMin meet')
              .attr('preserveAspectRatio', 'none')
              .attr('viewBox', dimensions)
              .classed('svg-content', true);

      var currentStep = '0'

      var offset_l = 50
      var offset_t = 10

      this.progressBar = svg.append('g')
                        .attr('transform', 'translate(' + offset_l + ',' + offset_t + ')')
                        .style('pointer-events', 'none')

      this.progressBackground = this.progressBar.append('rect')
          .attr('fill', this.colors.lightGreen)
          .attr('height', 4)
          .attr('width', width - offset * 2)
          .attr('rx', 4)
          .attr('ry', 4);

      this.progress = this.progressBar.append('rect')
          .attr('fill', this.colors.green)
          .attr('height', 4)
          .attr('width', 0)
          .attr('rx', 4)
          .attr('ry', 4);

      this.progress.transition()
          .duration(1000)
          .attr('width', function(){
              var index = self.steps.indexOf(currentStep);
              return (index + 1) * stepWidth;
          });

      this.progressBar.selectAll('circle')
          .data(this.steps)
          .enter()
          .append('circle')
          .attr('id', function(d, i){ return 'step_' + i; })
          .attr('cx', function(d, i){
          console.log("StepWidth:"+ stepWidth)
                return i * stepWidth;
          })
          .attr('cy', 4)
          .attr('r', radius)
          .attr('fill', '#FFFFFF')
          .attr('stroke', this.colors.lightGreen)
          .attr('strokeWidth', 6)

      this.progressBar.selectAll('text')
          .data(this.steps)
          .enter()
          .append('text')
          .attr('id', function(d, i){ return 'label_' + i; })
          .attr('dx', function(d, i){ return i * stepWidth; })
          .attr('dy', 10)
          .attr('text-anchor', 'middle')
          .text(function(d, i) { return i + 1; })

      this.updateProgressBar("0");

      }
   }

    updateProgressBar(step_){
        if(this.setupHappened) {
        const self = this

        var stepWidth = this.stepWidth

        console.log("Updating progress bar")
        this.progress.transition()
            .duration(1000)
            .attr('fill', this.colors.green)
            .attr('width', function(){
                var index = self.steps.indexOf(step_);
                return (index) * stepWidth;
            });

        for(var i = 0; i < this.steps.length; i++){

            if(i <= this.steps.indexOf(step_)) {

                d3.select('#step_' + i).attr('fill', this.colors.green).attr('stroke', this.colors.green);
                d3.select('#label_' + i).attr('fill', '#FFFFFF');


            } else {

                d3.select('#step_' + i).attr('fill', '#FFFFFF').attr('stroke', this.colors.lightGreen);
                d3.select('#label_' + i).attr('fill', '#000000');

            }

        }
      }
    }

   render() {
      return <div></div>
   }
}

const timeboxesReactElement = <TimeboxProgressBar />;

const tableDOMContainer = document.getElementById('timeboxes_container');
ag_global_vars.timeboxesRef = ReactDOM.render(timeboxesReactElement, tableDOMContainer);