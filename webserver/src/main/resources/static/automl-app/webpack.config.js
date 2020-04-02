const path = require('path');
const HtmlWebPackPlugin = require("html-webpack-plugin");
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = {
  mode: 'development',
  entry: './src/index.js',
  output: {
    filename: 'main.js',
    path: path.resolve(__dirname, 'dist'),
  },
  module: {
    rules: [
      {
        test: /\.(js|jsx)$/,
        exclude: /node_modules/,
        use: {
          loader: "babel-loader"
        }
      },
      {
        test: /\.html$/,
        use: [
          {
            loader: "html-loader"
          }
        ]
      },
      {
        test: /\.css$/,
        use: [
          MiniCssExtractPlugin.loader, // https://github.com/webpack-contrib/mini-css-extract-plugin
          {
            loader: 'css-loader',  // https://github.com/webpack-contrib/css-loader
            options: {
              importLoaders: 1,
              modules: {
                          mode: 'local',
                          localIdentName: '[path][name]__[local]--[hash:base64:5]',
                          context: path.resolve(__dirname, 'src'),
                          hashPrefix: 'my-custom-hash',
                        },
              modules: true
            }
          }
        ]
      }
    ]
  },
  plugins: [
    new MiniCssExtractPlugin(),
    new HtmlWebPackPlugin({
      template: "./src/index.html",
      filename: "./index.html"
    })
  ]
};
