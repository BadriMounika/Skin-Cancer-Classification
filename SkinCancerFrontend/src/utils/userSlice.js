import { createSlice } from "@reduxjs/toolkit";

const initialState = {
  name: "",
  emailId: "",
  role: "",
  loading: true,
};

const userSlice = createSlice({
  name: "user",
  initialState,
  reducers: {
    addUser: (state, action) => {
      state.name = action.payload.name;
      state.emailId = action.payload.emailId;
      state.role = action.payload.role;
      state.loading = false;
    },
    removeUser: (state) => {
      state.name = "";
      state.emailId = "";
      state.role = "";
      state.loading = false;
    },
    setLoadingTrue: (state) => {
      state.loading = false;
    },
  },
});

export const { addUser, removeUser, setLoadingTrue } = userSlice.actions;
export default userSlice.reducer;
