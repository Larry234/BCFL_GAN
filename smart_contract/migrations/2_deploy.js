const weights = artifacts.require("NetworkWeights");

module.exports = function (deployer) {
  deployer.deploy(weights);
};
