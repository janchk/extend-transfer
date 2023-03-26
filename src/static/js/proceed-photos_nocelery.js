// basic-upload.js

{/* <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script> */}

$(function () {
    $("#js-proceed-photos").click(function (event) {
        event.preventDefault(); // Prevent default form submission
        var uniqueId = generateUniqueId();
        setCookie("id", uniqueId, 1);
        var form = $(this).closest("form");
        var formData = new FormData(form[0]); // Collect form data
        formData.append('img_process_sync', "1"); // Add additional data
        $("#modal-processing").modal("show");

        $.ajax({
            type: 'post',
            // url: '{% url "mix" %}',
            url: '/mix',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                    $("#modal-processing").modal("hide");
                    window.location.href = '/mix';
            }
        });
    })
});
