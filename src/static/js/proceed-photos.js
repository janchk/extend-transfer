// basic-upload.js

{/* <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script> */}

$(function () {
    $("#js-proceed-photos").click(function (event) {
        event.preventDefault(); // Prevent default form submission
        var uniqueId = generateUniqueId();
        setCookie("id", uniqueId, 1);
        var form = $(this).closest("form");
        var formData = new FormData(form[0]); // Collect form data
        formData.append('img_to_process', "1"); // Add additional data

        $.ajax({
            type: 'post',
            url: '{% url "mix" %}',
            data: formData,
            processData: false,
            contentType: false,
            success: function (data) {
                $("#modal-progress").modal("show");
                requests = setInterval(function () {
                    $.ajax({
                        type: 'post',
                        url: '{% url "mix" %}',
                        data: {
                            "on_progress": "TRUE",
                            "tsk_id": data.tsk_id,
                            csrfmiddlewaretoken: "{{ csrf_token }}"
                        },
                        success: function (rspns) {
                            $(".progress-bar").css({
                                "width": rspns.progr
                            });
                            $(".progress-bar").text(rspns.progr);
                            console.log(rspns.progr, "and", rspns.sts)
                            if (rspns.sts == "SUCCESS") {
                                clearInterval(requests);
                                $("#modal-progress").modal("hide");
                                $("#task_id").val(data.tsk_id);
                                $("#result-load").submit();
                            }
                        }
                    })
                }, 1000)
            }
        });
    })
});
