const Admin = require("../models/admin");
const User = require("../models/user");
const { validateUser } = require("../utils/validation");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");

const generateToken = (adminId, role) => {
  return jwt.sign({ id: adminId, role }, process.env.JWT_SECRET, {
    expiresIn: "7d",
  });
};

const setTokenCookie = (res, token) => {
  res.cookie("token", token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === "production",
    sameSite: "strict",
    maxAge: 7 * 24 * 60 * 60 * 1000,
  });
};

exports.register = async (req, res) => {
  try {
    const { fullName, username, email, password, mobileNumber } = req.body;

    const { error } = validateUser(req.body);
    if (error) return res.status(400).json({ error: error.details[0].message });

    const adminExists = await Admin.findOne({
      $or: [{ email }, { username }, { mobileNumber }],
    });
    if (adminExists)
      return res.status(400).json({ error: "Admin already exists" });

    const hashedPassword = await bcrypt.hash(password, 10);

    const admin = new Admin({
      fullName,
      username,
      email,
      password: hashedPassword,
      mobileNumber,
      role: "user",
    });
    const savedAdmin = await admin.save();

    const token = generateToken(savedAdmin._id, savedAdmin.role);
    setTokenCookie(res, token);

    res.status(201).json({
      message: "User registered successfully",
      user: {
        id: savedAdmin._id,
        fullName,
        username,
        email,
        mobileNumber,
        role: savedAdmin.role,
      },
    });
  } catch (err) {
    console.error("Registration Error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};

exports.login = async (req, res) => {
  try {
    const { username, password } = req.body;
    const admin = await Admin.findOne({ username });
    if (!admin) return res.status(400).json({ error: "Invalid credentials" });

    const isMatch = await bcrypt.compare(password, admin.password);
    if (!isMatch) return res.status(400).json({ error: "Invalid credentials" });

    const token = generateToken(admin._id, admin.role);
    setTokenCookie(res, token);

    res.json({
      message: "Login successful",
      user: {
        id: admin._id,
        fullName: admin.fullName,
        username,
        email: admin.email,
        mobileNumber: admin.mobileNumber,
        role: admin.role,
      },
    });
  } catch (err) {
    console.error("Login Error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};


exports.getUsers = async (req, res) => {
  try {
    const users = await User.find().select("-password"); // Exclude password field
    res.json(users);
  } catch (err) {
    console.error("Get Users Error:", err);
    res.status(500).json({ error: "Internal server error" });
  }
};
