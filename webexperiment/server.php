<?php
// 设置文件类型和字符编码
header('Content-Type: text/html; charset=UTF-8');

// 跨域支持（可选）
header("Access-Control-Allow-Origin: *");

// 本地服务器端口（根据需要更改）
$port = 8000;

// 创建一个简单的PHP本地服务器
if (php_sapi_name() == 'cli-server') {
    $url = parse_url($_SERVER['REQUEST_URI']);
    $path = __DIR__ . $url['path'];
    
    if (is_file($path)) {
        return false;
    }
}

// 启动PHP内置服务器
$publicPath = __DIR__;
chdir($publicPath);
exec("php -S localhost:$port");