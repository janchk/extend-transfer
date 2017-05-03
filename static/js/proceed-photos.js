// basic-upload.js

$(function () {

  $(".js-proceed-photos").click(function () {
    $("#fileupload").click(); // for id = "fileupload"
    // go to the next page maybe
  });

  $("#fileupload").fileupload({
    dataType: 'json',
  //   // sequentialUploads: true,  /* 1. SEND THE FILES ONE BY ONE */
    start: function () {  /* 2. WHEN THE UPLOADING PROCESS STARTS, SHOW THE MODAL */
      $("#modal-progress").modal("show");
    },
    stop: function () {  /* 3. WHEN THE UPLOADING PROCESS FINALIZE, HIDE THE MODAL */
      $("#modal-progress").modal("hide");

    },
    progressall: function (strProgress) {  /* 4. UPDATE THE PROGRESS BAR */
      // var progress = parseInt(data.loaded / data.total * 100, 10);
      // var strProgress = progress + "%";
      $(".progress-bar").css({"width": strProgress});
      $(".progress-bar").text(strProgress);
    },
    done: function (e, data) {
  //     // if (data.result.is_valid) {
  //     //   $("#gallery tbody").prepend(
  //     //     "<tr><td><a href='" + data.result.url + "'>" + data.result.name + "</a></td></tr>"
  //       // )
  //     // }
    }
  //
  });

});