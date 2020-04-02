import React, { Component } from "react";
import ReactDOM from "react-dom";

import styles from './linechart.css';
import * as d3 from "d3"
import * as d3time from "d3-time"
import { scaleLinear } from 'd3-scale'
import { timeFormat } from "d3-time-format"

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
        var margin = { top: 30, right: 132, bottom: 30, left: 50 },
                width = 960 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom;

        var parseDate = d3.timeFormat("%m/%e/%Y"),
            bisectDate = d3.bisector(function(d) { return d.date; }).left,
            formatValue = d3.format(","),
            dateFormatter = d3.timeFormat("%m/%d/%y");

        var x = d3.scaleLinear()
                .range([0, width]);

        var y = d3.scaleLinear()
                .range([height, 0]);

        var xAxis = d3.axisBottom(x)
            .tickFormat(dateFormatter);

        var yAxis = d3.axisLeft(y)
            .tickFormat(d3.format("s"))

        var line = d3.line()
            .x(function(d) { return x(d.date); })
            .y(function(d) { return y(d.likes); });

        var svg = d3.select("#linechart_svg")
            .attr("width", width + margin.left + margin.right)
            .attr("height", height + margin.top + margin.bottom)
            .append("g")
            .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

        var tooltip = d3.select("#line_chart_container").append("div")
            .attr("class", "tooltip")
            .style("display", "none");



        d3.csv("data.csv").then(function( data) {

            data.forEach(function(d) {
                console.log("Linechart data point:" + JSON.stringify(d))
              });


            data.forEach(function(d) {
                d.date = parseDate(d.date);
                d.likes = +d.likes;
            });

            data.sort(function(a, b) {
                return a.date - b.date;
            });

            x.domain([data[0].date, data[data.length - 1].date]);
            y.domain(d3.extent(data, function(d) { return d.likes; }));

            svg.append("g")
                .attr("class", "x axis")
                .attr("transform", "translate(0," + height + ")")
                .call(xAxis);

            svg.append("g")
                .attr("class", "y axis")
                .call(yAxis)
                .append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 6)
                .attr("dy", ".71em")
                .style("text-anchor", "end")
                .text("Number of Likes");

            svg.append("path")
                .datum(data)
                .attr("class", "line")
                .attr("d", line);

            var focus = svg.append("g")
                .attr("class", "focus")
                .style("display", "none");

            focus.append("circle")
                .attr("r", 5);

            var tooltipDate = tooltip.append("div")
                .attr("class", "tooltip-date");

            var tooltipLikes = tooltip.append("div");
            tooltipLikes.append("span")
                .attr("class", "tooltip-title")
                .text("Likes: ");

            var tooltipLikesValue = tooltipLikes.append("span")
                .attr("class", "tooltip-likes");

            svg.append("rect")
                .attr("class", "overlay")
                .attr("width", width)
                .attr("height", height)
                .on("mouseover", function() { focus.style("display", null); tooltip.style("display", null);  })
                .on("mouseout", function() { focus.style("display", "none"); tooltip.style("display", "none"); })
                .on("mousemove", mousemove);


            function mousemove() {
                var x0 = x.invert(d3.mouse(this)[0]),
                    i = bisectDate(data, x0, 1),
                    d0 = data[i - 1],
                    d1 = data[i],
                    d = x0 - d0.date > d1.date - x0 ? d1 : d0;
                console.log("Mouse over" + x0)
                focus.attr("transform", "translate(" + x(d.date) + "," + y(d.likes) + ")");
                tooltip.attr("style", "left:" + (x(d.date) + 64) + "px;top:" + y(d.likes) + "px;");
                tooltip.select(".tooltip-date").text(dateFormatter(d.date));
                tooltip.select(".tooltip-likes").text(formatValue(d.likes));
            }
        });

   }

   render() {
        return <svg id="linechart_svg"> </svg>
   }
}

const tableElement = <Linechart/>;

const tableDOMContainer = document.getElementById('linechart_container');
ag_global_vars.linechartRef = ReactDOM.render(tableElement, tableDOMContainer);
