const express = require("express");
const {adminAuth} = require("../middlewares/auth");
const { register, login,getUsers } = require("../controllers/admin");

const router = express.Router();

// Register User Route
router.post("/register", register);
router.post("/login", login);
router.get("/users",getUsers);

module.exports = router;
