import type { NextApiRequest, NextApiResponse } from "next";
import axios from "axios";

export default axios.create({
  baseURL: "https://subme-api.herokuapp.com/",
});
