$(document).ready(function() {
    loadUsers();

    function loadUsers() {
        $.ajax({
            url: '/api/admin/users',
            method: 'GET',
            success: function(data) {
                var tableBody = $('#usersTable tbody');
                tableBody.empty();

                $.each(data, function(index, user) {
                    var row = '<tr>' +
                        '<td>' + user.firstName + '</td>' +
                        '<td>' + user.lastName + '</td>' +
                        '<td>' + user.email + '</td>' +
                        '<td>' +
                        '<select name="role-' + user.id + '" class="custom-select">' +
                        '<option value="ROLE_USER" ' + (user.role === 'ROLE_USER' ? 'selected' : '') + '>User</option>' +
                        '<option value="ROLE_ADMIN" ' + (user.role === 'ROLE_ADMIN' ? 'selected' : '') + '>Admin</option>' +
                        '</select>' +
                        '</td>' +
                        '<td>' +
                        '<button class="btn-primary_role" data-user-id="' + user.id + '">Изменить роль</button>' +
                        '<button class="btn-delete_user" data-user-id="' + user.id + '">Удалить</button>' +
                        '</td>' +
                        '</tr>';
                    tableBody.append(row);
                });

                $('.btn-primary_role').on('click', function() {
                    var userId = $(this).data('user-id');
                    var newRole = $('select[name="role-' + userId + '"]').val();
                    updateUserRole(userId, newRole);
                });

                $('.btn-delete_user').on('click', function() {
                    var userId = $(this).data('user-id');
                    deleteUser(userId);
                });
            },
            error: function() {
                alert('Ошибка при загрузке пользователей.');
            }
        });
    }

    function updateUserRole(userId, newRole) {
        $.ajax({                url: 'api/admin/users/update',
            method: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ id: userId, role: newRole }),
            success: function() {
                alert('Роль пользователя обновлена.');
                loadUsers();
            },
            error: function() {
                alert('Ошибка при обновлении роли пользователя.');
            }
        });
    }

    function deleteUser(userId) {
        if (confirm('Вы уверены, что хотите удалить этого пользователя?')) {
            $.ajax({
                url: '/api/admin/users/' + userId,
                method: 'DELETE',
                success: function() {
                    alert('Пользователь удален.');
                    loadUsers();
                },
                error: function() {
                    alert('Ошибка при удалении пользователя.');
                }
            });
        }
    }
});