$(function () {
    var $dropzone = $('.image_picker'),
        $droptarget = $('.drop_target'),
        $dropinput = $('#inputFile'),
        $dropimg = $('.image_preview'),
        $remover = $('[data-action="remove_current_image"]'),

        $dropzone_style = $('.image_picker_style'),
        $droptarget_style = $('.drop_target_style'),
        $dropinput_style = $('#inputFileStyle'),
        $dropimg_style = $('.image_preview_style'),
        $remover_style = $('[data-action="remove_current_image_style"]');

    $dropzone_style.on('dragover', function () {
        $droptarget_style.addClass('dropping');
        return false;
    });

    $dropzone.on('dragover', function () {
        $droptarget.addClass('dropping');
        return false;
    });

    $dropzone_style.on('dragend dragleave', function () {
        $droptarget.removeClass('dropping');
        return false;
    });

    $dropzone.on('dragend dragleave', function () {
        $droptarget.removeClass('dropping');
        return false;
    });


    $dropzone.on('drop', function (e) {
        $droptarget.removeClass('dropping');
        $droptarget.addClass('dropped');
        $remover.removeClass('disabled');
        e.preventDefault();

        var file = e.originalEvent.dataTransfer.files[0],
            reader = new FileReader();

        reader.onload = function (event) {
            $dropimg.css('background-image', 'url(' + event.target.result + ')');
        };

        console.log(file);
        reader.readAsDataURL(file);

        return false;
    });

    $dropzone_style.on('drop', function (e) {
        $droptarget_style.removeClass('dropping');
        $droptarget_style.addClass('dropped');
        $remover_style.removeClass('disabled');
        e.preventDefault();

        var file = e.originalEvent.dataTransfer.files[0],
            reader = new FileReader();

        reader.onload = function (event) {
            $dropimg_style.css('background-image', 'url(' + event.target.result + ')');
        };

        console.log(file);
        reader.readAsDataURL(file);

        return false;
    });


    $dropinput.change(function (e) {
        $droptarget.addClass('dropped');
        $remover.removeClass('disabled');
        // $('.image_title input').val('');

        var file = $dropinput.get(0).files[0],
            reader = new FileReader();

        reader.onload = function (event) {
            $dropimg.css('background-image', 'url(' + event.target.result + ')');
        }

        reader.readAsDataURL(file);
    });

    $dropinput_style.change(function (e) {
        $droptarget_style.addClass('dropped');
        $remover_style.removeClass('disabled');
        // $('.image_title input').val('');

        var file = $dropinput_style.get(0).files[0],
            reader = new FileReader();

        reader.onload = function (event) {
            $dropimg_style.css('background-image', 'url(' + event.target.result + ')');
        }

        reader.readAsDataURL(file);
    });

    $remover_style.on('click', function () {
        $dropimg_style.css('background-image', '');
        $droptarget_style.removeClass('dropped');
        $remover_style.addClass('disabled');
        // $('.image_title input').val('');
    });

    $remover.on('click', function () {
        $dropimg.css('background-image', '');
        $droptarget.removeClass('dropped');
        $remover.addClass('disabled');
        // $('.image_title input').val('');
    });

    // $('.image_title input').blur(function () {
    //     if ($(this).val() != '') {
    //         $droptarget.removeClass('dropped');
    //     }
    // });
    //
    // $('.image_title input').blur(function () {
    //     if ($(this).val() != '') {
    //         $droptarget_style.removeClass('dropped');
    //     }
    // });
});
