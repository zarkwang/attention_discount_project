

var clickedButtonIds = [];
var maxAmount = 100;

// Function to generate the price list rows
function generatePriceList(frontAmount, backAmount, seqLength, cond) {

    var sequenceText = "Receive £" + frontAmount + " today and £" + backAmount + " in " + seqLength

    var table = document.getElementById("priceListTable");
    for (var x = 0; x <= maxAmount; x++) {
        var row = document.createElement("tr");

        if (x === 0) {
            var optionACell = document.createElement("td");
            optionACell.textContent = sequenceText;
            optionACell.rowSpan = maxAmount +1;
            row.appendChild(optionACell);
        }

        var choiceCell = document.createElement("td");
        var radioContainer = document.createElement("div");
        radioContainer.className = "radio-container";

        var optionARadio = document.createElement("input");
        optionARadio.type = "radio";
        optionARadio.name = "choice_" + x;
        optionARadio.value = "Option A";
        optionARadio.id = "optionAButton_" + x;
        optionARadio.addEventListener("click", function () {
            clickedButtonIds.push(this.id);
            //updateClickedButtonsList();

            // Automatically check "Option A" buttons above and "Option B" buttons below
            currentRow = parseInt(this.id.split("_")[1]);
            for (var i = 0; i < currentRow; i++) {
                var optionABtn = document.getElementById("optionAButton_" + i);
                if (optionABtn) {
                    optionABtn.checked = true;
                }
            }
            for (var j = currentRow; j <= maxAmount; j++) {
                var optionBBtn = document.getElementById("optionBButton_" + j);
                if (optionBBtn) {
                    optionBBtn.checked = true;
                }
            }
            // Automatically click the "Show/Hide Rows" button
            if (currentRow % 10 == 0 && currentRow > 0){
                toggleHiddenRows(currentRow-9,currentRow-1)
            };

            updateSwtichRow();
        });

        var optionBRadio = document.createElement("input");
        optionBRadio.type = "radio";
        optionBRadio.name = "choice_" + x;
        optionBRadio.value = "Option B";
        optionBRadio.id = "optionBButton_" + x;
        optionBRadio.addEventListener("click", function () {
            clickedButtonIds.push(this.id);
            //updateClickedButtonsList();

            // Automatically check "Option A" buttons above and "Option B" buttons below
            currentRow = parseInt(this.id.split("_")[1]);
            for (var i = 0; i < currentRow; i++) {
                var optionABtn = document.getElementById("optionAButton_" + i);
                if (optionABtn) {
                    optionABtn.checked = true;
                }
            }
            for (var j = currentRow; j <= maxAmount; j++) {
                var optionBBtn = document.getElementById("optionBButton_" + j);
                if (optionBBtn) {
                    optionBBtn.checked = true;
                }
            }
            // Automatically click the "Show/Hide Rows" button
            if (currentRow % 10 == 0 && currentRow > 0){
                toggleHiddenRows(currentRow-9,currentRow-1)
            };

            updateSwtichRow();
        });

        var optionAButton = document.createElement("label");
        optionAButton.textContent = "Option A";
        optionAButton.htmlFor = "optionAButton_" + x;

        var optionBButton = document.createElement("label");
        optionBButton.textContent = "Option B";
        optionBButton.htmlFor = "optionBButton_" + x;

        radioContainer.appendChild(optionARadio);
        radioContainer.appendChild(optionAButton);
        radioContainer.appendChild(optionBRadio);
        radioContainer.appendChild(optionBButton);
        choiceCell.appendChild(radioContainer);
        row.appendChild(choiceCell);

        var optionBCell = document.createElement("td");
        if (cond === "front-align"){
            optionBCell.textContent = "Receive £" + x + " today";
        } else if (cond === "back-align") {
            optionBCell.textContent = "Receive £" + x + " in " + seqLength;
        }
        
        row.appendChild(optionBCell);

        table.appendChild(row);
    }
};


// Function to show hidden rows within a range
function toggleHiddenRows(start, end) {
    hideRowsWithNonIntegerCondition();

    var table = document.getElementById("priceListTable");
    for (var x = start; x <= end; x++) {
        var row = table.getElementsByTagName('tr')[x];
        if (row) {
            var isHidden = row.style.display === "none";
            row.style.display = isHidden ? "table-row" : "none";
            
            // Un-check the radios
            var radios = row.querySelectorAll("input[type='radio']");
            radios.forEach(function (radio) {
                radio.checked = false;
            });
        }   
    }
}

// Function to initially hide rows where x/10 is not an integer
function hideRowsWithNonIntegerCondition() {
    for (var x = 1; x <= maxAmount; x++) {
        if ((x-1) % 10 !== 0) {
            var rowToHide = document.getElementsByTagName("tr")[x];
            if (rowToHide) {
                rowToHide.style.display = "none";
            }
        }
    }
}


// Show the swtich point: indifference point of option B for option A
function updateSwtichRow() {
    var switchRow = document.getElementById("switchRow");    
    if (switchRow) {
        switchRow.innerHTML = currentRow;
    }
}


// Function to update the list of clicked button IDs
// function updateClickedButtonsList() {
//     var clickedButtonsList = document.getElementById("clickedButtonsList");
//     clickedButtonsList.innerHTML = "";
//     clickedButtonIds.forEach(function (id) {
//         var listItem = document.createElement("li");
//         listItem.textContent = id;
//         clickedButtonsList.appendChild(listItem);
//     });
// }


// Dynamically create buttons for specific ranges
// var buttonContainer = document.getElementById("buttonContainer");
// for (var i = 1; i <= maxAmount; i += 10) {
//     let start = i;
//     let end = i + 8;
//     var button = document.createElement("button");
//     button.id = "showHideButton_" + start + "-" + end; // Unique ID for each button
//     button.textContent = "Show/Hide Rows " + start + "-" + end;
//     button.addEventListener("click", function () {
//         toggleHiddenRows(start, end);
//     });
//     buttonContainer.appendChild(button);
// };


// Call the function to generate the price list





