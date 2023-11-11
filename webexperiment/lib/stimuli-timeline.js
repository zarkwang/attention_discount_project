
const stimuli = [
    {"front_amount": 30, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"},
    {"front_amount": 50, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"},
    {"front_amount": 70, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"},
    {"front_amount": 90, "backend_amount": 30, "seq_length": "3 months", "condition": "front-align"}
    ]



const pageIteration = ['intertemporal-choice','confidence-check']

const intertemporalChoicePage = `
    <p>Which option would you prefer in each row?</p>
    <table>
        <thead>
            <tr>
                <th>Option A</th>
                <th>Choice</th>
                <th>Option B</th>
            </tr>
        </thead>
        <tbody id="priceListTable">
            <!-- Rows will be generated here using JavaScript -->
        </tbody>
    </table>

    <div id="switchRowContainer">
        <h2>Switch Row:<span id="switchRow"></span></h2>
    </div>
`

const configConfidenceCheck = {
    "elements": [
      {
        "type": "radiogroup",
        "name": "confidence_rate",
        "title": "To what extent are you sure about your choice?",
        "isRequired": true,
        "colCount": 1,
        "choices": [ "Totally not sure", 
                     "Slightly sure", 
                     "Moderately sure", 
                     "Quite sure", 
                     "Absolutely sure"],
        "separateSpecialChoices": true,
        "showClearButton": true
      }
    ],
    "showQuestionNumbers": false,
    "showNavigationButtons": false,
    "width": "1000px"
  };
