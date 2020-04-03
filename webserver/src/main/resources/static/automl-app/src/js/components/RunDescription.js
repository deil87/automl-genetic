import React, { Component } from 'react'
import ReactDOM from "react-dom"

class RunDescription extends Component {
   constructor(props){
      super(props)
      this.changeState = this.changeState.bind(this);
      this.state = {
            counter: 33
          };
   }

   componentDidMount() {
//      this.createTBProgressBar()
   }
   componentDidUpdate() {
    console.log("Property server was updated to(componentDidUpdate):")
   }

   changeState() {
     console.log("Property server was updated to(changeState):");
   };

   render() {
      return <dl className="row">
                 <dt className="col-sm-3">{this.state.counter}</dt>
                 <dd className="col-sm-9">A description list is perfect for defining terms.</dd>

                 <dt className="col-sm-3">Euismod</dt>
                 <dd className="col-sm-9">
                     <p>Vestibulum id ligula porta felis euismod semper eget lacinia odio sem nec elit.</p>
                     <p>Donec id elit non mi porta gravida at eget metus.</p>
                 </dd>
                 <dt className="col-sm-3">Nesting</dt>
                 <dd className="col-sm-9">
                     <dl className="row">
                         <dt className="col-sm-4">Nested definition list</dt>
                         <dd className="col-sm-8">Aenean posuere, tortor sed cursus feugiat, nunc augue blandit nunc.</dd>
                     </dl>
                 </dd>
             </dl>
   }
}

export default RunDescription;