import _ from 'lodash';
import Form from "./js/components/Form";
import Barchart from "./js/components/Barchart";

function component() {
  const element = document.createElement('div');

  element.innerHTML = _.join(['Hello', 'webpack in AutoML'], ' ');

  return element;
}

document.body.appendChild(component());
