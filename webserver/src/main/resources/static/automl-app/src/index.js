import _ from 'lodash';
import Form from "./js/components/Form";
import Barchart from "./js/components/Barchart";
import Table from "./js/components/Table";

function component() {
  const element = document.createElement('div');

  element.innerHTML = _.join(['Hello', 'webpack in AutoML'], ' ');

  return element;
}

document.body.appendChild(component());
