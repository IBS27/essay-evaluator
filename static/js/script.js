function countChar(essayNum) {
  var textarea = document.getElementById("essay" + essayNum);
  var characterCount = document.getElementById("count" + essayNum);
  characterCount.setAttribute("data-count", textarea.value.length);

  // const currentLength = textarea.value.length;
  // const current = $("#current" + essayNum);

  // if (currentLength < 1000) {
  //   current.css("color", "#C93B33");
  // } else {
  //   current.css("color", "#5AB656");
  // }
}

function confirmSubmission() {
  var confirmation = confirm("Are you sure you want to submit?");
  return confirmation;
}
